import os
import numpy as np
import torch
import pickle 
from sentence_transformers import SentenceTransformer
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict
from transformers import AutoTokenizer, BertModel
from kmeans_pytorch import kmeans
import clip
import argparse

__DATA_DIR = os.getcwd()
__ANNOTATION_DIR = None


def getUniqueActions50Salads(root_dir:str) -> str:
    """
    Get unique action lists from annotation files on 50Salads dataset

    Args:
        root_dir:str -> path to annotations directory
    
    Return:
        actions:set(str) -> Unique actions found in the dataset

    """

    annotation_files = os.listdir(root_dir) 
    actions = set()
    for file_ in annotation_files:
        f = open(os.path.join(root_dir, file_), "r")
        content = f.readlines()
        for line in content:
            action = line.split()[-1]
            for ending in ["prep", "core", "post"]:
                if ending in action.split("_"):
                    action = action.replace(ending, "")
            action = action.replace("_", " ")
            actions.add(action)
    
    return actions


def ActionLanguageEncoding(actions:str, save_path:str):
    """
    Encodes action using a LLM

    Args:
        actions:list[str] -> List of actions
        save_path:str -> path to save encodings
    
    Returns:
        ActionEncodingDict:dict[str, float] -> Dictionary containing action name and its absolute representation.
    
    """

    #build encoder
    encoder = SentenceTransformer("stsb-mpnet-base-v2")
    #encode actions
    ActionEncodingDict = {}
    for action in actions:
        encoding = encoder.encode(action)
        ActionEncodingDict[action] = encoding

    return ActionEncodingDict


def ActionLanguageEncodingCLIP(actions:str, save_path:str):
    """
    Encodes action using a LLM

    Args:
        actions:list[str] -> List of actions
        save_path:str -> path to save encodings
    
    Returns:
        ActionEncodingDict:dict[str, float] -> Dictionary containing action name and its absolute representation.
    
    """

    #build encoder
    # encoder = SentenceTransformer("stsb-mpnet-base-v2")
    encoder, preprocess = clip.load("ViT-B/16", device='cpu')

    #encpde actions
    ActionEncodingDict = {}
    for action in actions:
        action_tokens = clip.tokenize(action).to('cpu')
        encoding = encoder.encode_text(action_tokens)
        ActionEncodingDict[action] = encoding.squeeze()

    return ActionEncodingDict


def GetSimilarities(encodings:torch.Tensor)->torch.Tensor:
    """
    Calculate similarities between encodings

    Args:
        actions:str -> list of actions
        encodings:list[torch.Tensor] -> list of encodings
    
    Return:
        similarities:torch.Tensor -> similarity scores
    """
    # similarities = []
    cosine = torch.nn.CosineSimilarity(dim=2, eps=1e-08)
    similarities = cosine(encodings.unsqueeze(dim=1), encodings.unsqueeze(dim=0))
    return similarities



def getClassActionEgtea(file_path):
    """
    Args:
        file_path (str): Path to action.csv file
    Return:
        out_action (dict): Dictionaries of action class to action name mapping 
    """

    file_ = open(file=file_path, mode="r")
    lines = file_.readlines()
    out_actions = {}
    out_verbs = {}
    out_nouns = {}

    for line in lines:

        line = line.split(",")
        action = line[0]
        verb_noun = line[1].split("_")
        verb, noun = verb_noun[0], verb_noun[1]
        text_action = line[-1].strip("\n").split("_")
        text_verb = text_action[0].split("/")[0][1:] #Removes the blank space before the verb
        text_noun = text_action[1].replace(":", " ")
        text_action = " ".join([text_verb, text_noun])

        out_actions[action] = text_action
        out_verbs[action] = text_verb
        out_nouns[action] = text_noun
    
    return out_actions, out_nouns, out_verbs

def ActionLanguageEncodingEgtea(actions, save_path="./encodings_egtea.pth"):
    """
    Encodes action using a LLM

    Args:
        actions:list[str] -> List of actions
        save_path:str -> path to save encodings
    
    Returns:
        ActionEncodingDict:dict[str, float] -> Dictionary containing action name and its absolute representation.
    
    """

    #build encoder
    # encoder = SentenceTransformer("stsb-mpnet-base-v2")
    encoder = SentenceTransformer("all-mpnet-base-v2")

    #encpde actions
    EncodedDict = {}
    for action in actions:
        encoded_action = encoder.encode(actions[action], convert_to_tensor=True).to(torch.float32).cpu()
        EncodedDict[int(action)] = encoded_action
    torch.save(EncodedDict, save_path)
    return EncodedDict


def ActionLanguageEncodingEgteaCLIP(actions, save_path="./encodings_egtea.pth"):
    """
    Encodes action using a LLM

    Args:
        actions:list[str] -> List of actions
        save_path:str -> path to save encodings
    
    Returns:
        ActionEncodingDict:dict[str, float] -> Dictionary containing action name and its absolute representation.
    
    """

    #build encoder
    # encoder = SentenceTransformer("stsb-mpnet-base-v2")
    encoder, preprocess = clip.load("ViT-B/16", device='cpu')


    #encpde actions
    EncodedDict = {}
    for action in actions:
        action_tokens = clip.tokenize(actions[action]).cpu()
        with torch.no_grad():
            encoded_action = encoder.encode_text(action_tokens).cpu()
        EncodedDict[int(action)] = encoded_action.squeeze()
    torch.save(EncodedDict, save_path)
    return EncodedDict



def GetSimilaritiesEgtea(encodings_dict, save_path="./similarities_egtea.pth"):
    """
    Calculate similarities between encodings

    Args:
        encodings_dict:{action:encoding} -> dict of encodings
    
    Return:
        similarities:torch.Tensor -> similarity scores
    """
    # similarities = []
    cosine = torch.nn.CosineSimilarity(dim=2, eps=1e-08)
    encodings = torch.stack(list(encodings_dict.values()))
    encodings = encodings.unsqueeze(dim=0)
    similarities = {}
    print(encodings.shape)

    for action in encodings_dict:
        action_encoding = torch.tensor(encodings_dict[action].reshape(1, 1, -1))
        similarities_ = cosine(action_encoding, encodings)
        similarities[int(action)] = similarities_.cpu().to(torch.float32)
    torch.save(similarities, save_path)
    return similarities



def GetSimilaritiesEgteaCLIP(encodings_dict, save_path="./similarities_egtea.pth"):
    """
    Calculate similarities between encodings

    Args:
        encodings_dict:{action:encoding} -> dict of encodings
    
    Return:
        similarities:torch.Tensor -> similarity scores
    """
    cosine = torch.nn.CosineSimilarity(dim=2, eps=1e-08)
    encodings = torch.stack(list(encodings_dict.values()))
    encodings = encodings.unsqueeze(dim=0)
    similarities = {}
    for action in encodings_dict:
        action_encoding = torch.tensor(encodings_dict[action].reshape(1, 1, -1))
        similarities_ = cosine(action_encoding, encodings)
    torch.save(similarities, save_path)
    return similarities


def KMeans(path_to_encodings, num_clusters):
    """
    k-means clustering on a set of language embeddigs
    Args:
        encodings_dict (dict): dict containing encodings
        num_clusters (int): number of actual clusters to learn
        max_ters (int): number of iterations

    Returns:
        cluster_ids_x (torch.tensor): cluster idx for every point
        cluster_centers (torch.tensor): cluster centers
    """
    encodings_dict = torch.load(path_to_encodings)
    points = torch.tensor(list(encodings_dict.values()))
    print("points:", points)
    cluster_ids_x, cluster_centers = kmeans(X=points, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0'))  
    return cluster_ids_x, cluster_centers


def getClassActionEpic(file_path):
    """
    Args:
        file_path (str): Path to action.csv file
    Return:
        out_action (dict): Dictionaries of action class to action name mapping 
    """

    file_ = open(file=file_path, mode="r")
    lines = file_.readlines()
    out_actions = {}
    out_verbs = {}
    out_nouns = {}

    for line in lines:

        line = line.split(",")
        action = line[0]
        verb_noun = line[1].split("_")
        verb, noun = verb_noun[0], verb_noun[1]
        noun_splitted = noun.split(":")
        if len(noun_splitted) > 1:
            noun = " ".join([noun_splitted[-1], noun_splitted[0]]) 
        print(f"{action} {verb} {noun}")

        text_action = " ".join([verb, noun])

        out_actions[action] = text_action
        out_verbs[action] = verb
        out_nouns[action] = noun

    return out_actions, out_nouns, out_verbs


def getClassActionEpic100(file_path):
    """
    Args:
        file_path (str): Path to action.csv file
    Return:
        out_action (dict): Dictionaries of action class to action name mapping 
    """

    file_ = open(file=file_path, mode="r")
    lines = file_.readlines()
    out_actions = {}
    out_verbs = {}
    out_nouns = {}

    for line in lines:

        line = line.split(",")
        action = line[0]
        verb_noun = line[-1].strip("\n").split(" ")
        verb, noun = verb_noun[0], verb_noun[1]
        noun_splitted = noun.split(":")
        if len(noun_splitted) > 1:
            noun = " ".join([noun_splitted[-1], noun_splitted[0]]) 
        # print(f"{action} {verb} {noun}")

        text_action = " ".join([verb, noun])
        # print(text_action)

        out_actions[action] = text_action
        out_verbs[action] = verb
        out_nouns[action] = noun

    return out_actions, out_nouns, out_verbs

def getClassAction50Salads(file_path):
    """
    Args:
        file_path (str): Path to action.csv file
    Return:
        out_action (dict): Dictionaries of action class to action name mapping 
    """

    file_ = open(file=file_path, mode="r")
    lines = file_.readlines()
    out_actions = {}

    for line in lines:

        line = line.split(" ")
        action = line[0]
        action_text = line[1].replace("\n", "")
        action_text = action_text.replace("_", " ")
        # print(f"{action} : {action_text}")
        out_actions[action] = action_text

    return out_actions


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create bank of actions")
    parser.add_argument("--dataset", type=str, default="egtea", help="Dataset to use")
    parser.add_argument("--label_file", type=str, default="./actions.csv", help="Path to action file in path relative to cwd")

    args = parser.parse_args()

    if args.dataset != "epic100" and args.dataset != "epic55":
        __ANNOTATION_DIR = os.path.join(__DATA_DIR, args.label_file)
        actions, nouns, verbs = getClassActionEgtea(__ANNOTATION_DIR)

    elif args.dataset == "epic100":
        __ANNOTATION_DIR = os.path.join(__DATA_DIR, args.label_file)
        actions, nouns, verbs = getClassActionEpic100(__ANNOTATION_DIR)
    else:
        __ANNOTATION_DIR = os.path.join(__DATA_DIR, args.label_file)
        actions, nouns, verbs = getClassActionEpic(__ANNOTATION_DIR)

    encodings_path = f"./encodings_{args.dataset}.pth"
    similarities_path = f"./similarities_{args.dataset}.pth"

    Encoded_dict = ActionLanguageEncodingEgtea(actions, save_path=encodings_path)
    similarities = GetSimilaritiesEgtea(Encoded_dict, save_path=similarities_path)


    