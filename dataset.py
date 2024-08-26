import dgl
import numpy as np
import torch
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model.model_zoo import *
from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer, mol_to_bigraph
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import DebertaV2Tokenizer

morded_calculator = Calculator(descriptors, ignore_3D=False)


nbits = 1024
longbits = 16384
# RDKit descriptors -->
calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
# dictionary
fpFunc_dict = {}
fpFunc_dict['ecfp0'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits)
fpFunc_dict['ecfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits)
fpFunc_dict['ecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
fpFunc_dict['ecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
fpFunc_dict['fcfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(
    m, 1, useFeatures=True, nBits=nbits
)
fpFunc_dict['fcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(
    m, 2, useFeatures=True, nBits=nbits
)
fpFunc_dict['fcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(
    m, 3, useFeatures=True, nBits=nbits
)
fpFunc_dict['lecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=longbits)
fpFunc_dict['lecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=longbits)
fpFunc_dict['lfcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(
    m, 2, useFeatures=True, nBits=longbits
)
fpFunc_dict['lfcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(
    m, 3, useFeatures=True, nBits=longbits
)
fpFunc_dict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)  # 167
fpFunc_dict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
    m, nBits=nbits
)
fpFunc_dict['hashtt'] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
    m, nBits=nbits
)
fpFunc_dict['avalon'] = lambda m: fpAvalon.GetAvalonFP(m, nbits)
fpFunc_dict['laval'] = lambda m: fpAvalon.GetAvalonFP(m, longbits)
fpFunc_dict['rdk5'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2)
fpFunc_dict['rdk6'] = lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2)
fpFunc_dict['rdk7'] = lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2)
fpFunc_dict['rdkDes'] = lambda m: calc.CalcDescriptors(m)  # 208

MAX_SEQ_LEN = 256


def preprocess_function(text, tokenizer):
    all_input_ids = []
    all_input_mask = []

    for sentence in text:
        tokens = tokenizer.tokenize(sentence)

        # limit size to make room for special tokens
        if MAX_SEQ_LEN:
            tokens = tokens[0 : (MAX_SEQ_LEN - 2)]

        # add special tokens
        tokens = [tokenizer.cls_token, *tokens, tokenizer.sep_token]

        # convert tokens to IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # create mask same size of input
        input_mask = [1] * len(input_ids)

        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)

    # pad up to max length
    # up to max_seq_len if provided, otherwise the max of current batch
    max_length = MAX_SEQ_LEN

    all_input_ids = torch.LongTensor(
        [i + [tokenizer.pad_token_id] * (max_length - len(i)) for i in all_input_ids]
    )
    all_input_mask = torch.FloatTensor([m + [0] * (max_length - len(m)) for m in all_input_mask])
    # print(all_input_ids.size(), all_input_mask.size())
    return all_input_ids[0], all_input_mask[0]


CHAR_SMI_SET = {
    "(": 1,
    ".": 2,
    "0": 3,
    "2": 4,
    "4": 5,
    "6": 6,
    "8": 7,
    "@": 8,
    "B": 9,
    "D": 10,
    "F": 11,
    "H": 12,
    "L": 13,
    "N": 14,
    "P": 15,
    "R": 16,
    "T": 17,
    "V": 18,
    "Z": 19,
    "\\": 20,
    "b": 21,
    "d": 22,
    "f": 23,
    "h": 24,
    "l": 25,
    "n": 26,
    "r": 27,
    "t": 28,
    "#": 29,
    "%": 30,
    ")": 31,
    "+": 32,
    "-": 33,
    "/": 34,
    "1": 35,
    "3": 36,
    "5": 37,
    "7": 38,
    "9": 39,
    "=": 40,
    "A": 41,
    "C": 42,
    "E": 43,
    "G": 44,
    "I": 45,
    "K": 46,
    "M": 47,
    "O": 48,
    "S": 49,
    "U": 50,
    "W": 51,
    "Y": 52,
    "[": 53,
    "]": 54,
    "a": 55,
    "c": 56,
    "e": 57,
    "g": 58,
    "i": 59,
    "m": 60,
    "o": 61,
    "s": 62,
    "u": 63,
    "y": 64,
}


def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=np.int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch]

    return X


def mordred(smiles):
    return np.array(
        [list(morded_calculator(Chem.MolFromSmiles(smiles)).fill_missing(value=0.0).values())]
    ).astype(np.float32)


def collate(gs):
    return dgl.batch(gs)


def graph_infomax_fp(smiles, model):
    model.eval()
    graphs = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            g = mol_to_bigraph(
                mol,
                add_self_loop=True,
                node_featurizer=PretrainAtomFeaturizer(),
                edge_featurizer=PretrainBondFeaturizer(),
                canonical_atom_order=True,
            )
            graphs.append(g)

        except:
            continue

    data_loader = DataLoader(graphs, batch_size=1, collate_fn=collate, shuffle=False)

    readout = AvgPooling()

    mol_emb = []
    for batch_id, bg in enumerate(data_loader):
        bg = bg.to('cpu')
        nfeats = [
            bg.ndata.pop('atomic_number').to('cpu'),
            bg.ndata.pop('chirality_type').to('cpu'),
        ]
        efeats = [
            bg.edata.pop('bond_type').to('cpu'),
            bg.edata.pop('bond_direction_type').to('cpu'),
        ]
        with torch.no_grad():
            node_repr = model(bg, nfeats, efeats)
        mol_emb.append(readout(bg, node_repr))
    mol_emb = torch.cat(mol_emb, dim=0).detach().cpu().numpy()
    # print(mol_emb.shape)
    return mol_emb


class PPIMModalDataset(Dataset):
    def __init__(self, all_smiles, labels):
        self.all_smiles = all_smiles
        self.labels = labels
        self.length = len(self.labels)
        self.drug2vox = {}
        self.tokenizer = DebertaV2Tokenizer('./DeBERTA/pretrain/smi.model')
        self.drug2tokens = {}
        self.drug2hasap = {}
        self.drug2ecfp = {}
        self.drug2rdk = {}

        self._pre_process()

    def _pre_process(self):
        for smile in tqdm(self.all_smiles):
            token_outputs = preprocess_function([smile], self.tokenizer)
            self.drug2tokens[smile] = token_outputs

            mol = Chem.MolFromSmiles(smile)
            fp = fpFunc_dict['rdk7'](mol)      # rdkit
            self.drug2rdk[smile] = np.asarray(fp)
            fp = fpFunc_dict['ecfp4'](mol)      # ecfp
            self.drug2ecfp[smile] = np.asarray(fp)
            fp = fpFunc_dict['hashap'](mol)     # hashap
            self.drug2hasap[smile] = np.asarray(fp)

    def __getitem__(self, idx):
        label = self.labels[idx]
        drug_token = self.drug2tokens[self.all_smiles[idx]]
        hash_fp = self.drug2hasap[self.all_smiles[idx]]
        rdk_fp = self.drug2rdk[self.all_smiles[idx]]
        ecfp = self.drug2ecfp[self.all_smiles[idx]]

        return (
            drug_token[0],
            drug_token[1],
            torch.FloatTensor(hash_fp),
            torch.FloatTensor(rdk_fp),
            torch.FloatTensor(ecfp),
            torch.FloatTensor([label]),
        )

    def __len__(self):
        return self.length
