import copy
import os
import torch
import torch.nn as nn
from typing import List
from rdkit import Chem
from torchdrug import models, layers, data
from torchdrug.layers import geometry

from src.model.protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize, StructureDatasetPDB


# Workaround for avoiding the error: Attribute Error: xxx is not a torch.nn.Module
# For more details: https://github.com/DeepGraphLearning/torchdrug/issues/77
nn.Module = nn._Module


class ProteinMPNNStructureEncoder(nn.Module):
    def __init__(self, model_path: str):
        super(ProteinMPNNStructureEncoder, self).__init__()
        self.model_path = model_path
        self.hidden_dim = 128
        self.num_layers = 3
        self.num_letters = 21
        self.num_edges = 48
        self.ca_only = False
        self.model_checkpoint = torch.load(self.model_path, map_location='cpu')
        if 'model_state_dict' in self.model_checkpoint.keys():
            self.model_checkpoint = self.model_checkpoint['model_state_dict']
        self.model = ProteinMPNN(
            num_letters=self.num_letters, node_features=self.hidden_dim, edge_features=self.hidden_dim,
            hidden_dim=self.hidden_dim, num_encoder_layers=self.num_layers, num_decoder_layers=self.num_layers,
            k_neighbors=self.num_edges, augment_eps=0.0  # set augment_eps to 0.0 to disable randomness
        )
        new_model_checkpoint = {}
        for key in self.model.state_dict().keys():
            for checkpoint_key in self.model_checkpoint.keys():
                if key in checkpoint_key:
                    new_model_checkpoint[key] = self.model_checkpoint[checkpoint_key]
        self.model.load_state_dict(new_model_checkpoint)

    def forward(self, pdb_files: List[str]):
        model_device = next(self.model.parameters()).device
        structure_representations = []
        for pdb_file in pdb_files:
            pdb_dict_list = parse_PDB(pdb_file, ca_only=self.ca_only)
            dataset = StructureDatasetPDB(pdb_dict_list, max_length=200000)

            all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9] == 'seq_chain']  # ['A','B', 'C',...]
            designed_chain_list = all_chain_list
            fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
            chain_id_dict = {}
            chain_id_dict[pdb_dict_list[0]['name']] = (designed_chain_list, fixed_chain_list)

            protein = dataset[0]
            batch_clones = [copy.deepcopy(protein)]
            X, _, mask, lengths, _, chain_encoding_all, _, _, _, _, _, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(
                batch_clones, model_device, chain_id_dict, None, None, None, None, None, ca_only=self.ca_only
            )

            structure_representation = self.model(X, _, mask, _, residue_idx, chain_encoding_all, _)['node_representation']
            attention_mask = torch.ones(structure_representation.shape[1], device=model_device)

            structure_representations.append({
                'representation': structure_representation,
                'attention_mask': attention_mask
            })

        max_node_length = max([representation['representation'].shape[1] for representation in structure_representations])
        for representation in structure_representations:
            representation['representation'] = torch.cat(
                [representation['representation'],
                 torch.zeros(
                     (representation['representation'].shape[0], max_node_length - representation['representation'].shape[1], representation['representation'].shape[2]),
                     device=model_device
                 )],
                dim=1
            )
            representation['representation'] = representation['representation'].view(-1, self.hidden_dim)
            representation['attention_mask'] = torch.cat(
                [representation['attention_mask'],
                 torch.zeros(max_node_length - representation['attention_mask'].shape[0], device=model_device)],
                dim=0
            )
        structure_representations = {
            k: torch.stack([representation[k] for representation in structure_representations]).detach().cpu()
            for k in structure_representations[0].keys()
        }

        return structure_representations


class GearNetStructureEncoder(nn.Module):
    def __init__(self, model_path: str):
        super(GearNetStructureEncoder, self).__init__()
        self.model_path = model_path
        self.graph_construction_model = layers.GraphConstruction(
            node_layers=[geometry.AlphaCarbonNode()],
            edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                         geometry.KNNEdge(k=10, min_distance=5),
                         geometry.SequentialEdge(max_distance=2)],
            edge_feature='gearnet'
        )
        self.model_checkpoint = torch.load(self.model_path, map_location='cpu')
        self.model = models.GearNet(
            input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512],
            num_relation=7, edge_input_dim=59, num_angle_bin=8,
            batch_norm=True, concat_hidden=True, short_cut=True, readout='sum'
        )
        new_model_checkpoint = {}
        for key in self.model.state_dict().keys():
            for checkpoint_key in self.model_checkpoint.keys():
                if key in checkpoint_key:
                    new_model_checkpoint[key] = self.model_checkpoint[checkpoint_key]
                    break
        self.model.load_state_dict(new_model_checkpoint)

    @staticmethod
    def from_pdb(pdb_file, atom_feature="default", bond_feature="default", residue_feature="default",
                 mol_feature=None, kekulize=False):
        # Create a protein from a PDB file. (A reimplementation of torchdrug.data.Protein.from_pdb)
        if not os.path.exists(pdb_file):
            raise FileNotFoundError("No such file `%s`" % pdb_file)
        mol = Chem.MolFromPDBFile(pdb_file, sanitize=False, proximityBonding=True)
        if mol is None:
            raise ValueError("RDKit cannot read PDB file `%s`" % pdb_file)
        return data.Protein.from_molecule(mol, atom_feature, bond_feature, residue_feature, mol_feature, kekulize)

    def forward(self, pdb_files: List[str]):
        model_device = next(self.model.parameters()).device
        proteins = [
            self.from_pdb(pdb_file, atom_feature="position", bond_feature="length", residue_feature="symbol") for pdb_file in pdb_files
        ]
        proteins = data.Protein.pack(proteins).to(model_device)
        proteins = self.graph_construction_model(proteins)
        proteins.view = 'residue'
        residue_lengths = [protein.num_residue.detach().cpu().item() for protein in proteins]
        node_features = self.model(proteins, proteins.node_feature.float())['node_feature']

        structure_representations = []
        total_length = 0
        for i, length in enumerate(residue_lengths):
            node_feature = node_features[total_length: total_length + length]
            node_mask = torch.ones(node_feature.shape[0], device=model_device)
            total_length += length

            structure_representations.append({
                'representation': node_feature,
                'attention_mask': node_mask
            })

        max_node_length = max([representation['representation'].shape[0] for representation in structure_representations])
        for representation in structure_representations:
            representation['representation'] = torch.cat(
                [representation['representation'],
                 torch.zeros(
                     (max_node_length - representation['representation'].shape[0],
                      representation['representation'].shape[1]),
                     device=model_device
                 )],
                dim=0
            )
            representation['attention_mask'] = torch.cat(
                [representation['attention_mask'],
                 torch.zeros(max_node_length - representation['attention_mask'].shape[0], device=model_device)],
                dim=0
            )
        structure_representations = {
            k: torch.stack([representation[k] for representation in structure_representations]).detach().cpu()
            for k in structure_representations[0].keys()
        }

        return structure_representations
