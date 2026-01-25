import os
import time
import random
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from PIL import Image, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt
import Training_Config as cfg
from Training_Training_Loop import train_overfit
from Training_Training_Loop_Raw import train_raw
from Training_Training_Loop_Heatmap import train_heatmap

def main():
    print("\nTraining mode options:")
    print("1. Start fresh (new weights)(Overfitting)")
    print("2. Resume from initial weights(Overfitting)")
    print("3. Resume from final weights(Overfitting)")
    print("4. Resume from last checkpoint(Overfitting)")
    print("5. Start fresh (new weights)(PoseNet)")
    print("6. Resume from initial weights(PoseNet)")
    print("7. Resume from final weights(PoseNet)")
    print("8. Resume from last checkpoint(PoseNet)")
    print("9. Start fresh (new weights)(HeatmapPoseNet)")
    print("10. Resume from initial weights(HeatmapPoseNet)")
    print("11. Resume from final weights(HeatmapPoseNet)")
    print("12. Resume from last checkpoint(HeatmapPoseNet)")

    choice = input("Enter choice [1/2/3/4/5/6/7/8/9/10/11/12]: ").strip()

    if choice == "1":
        train_overfit(load_initial=False)
    elif choice == "2":
        train_overfit(load_initial=True)
    elif choice == "3":
        train_overfit(load_final=True)
    elif choice == "4":
        train_overfit(resume_checkpoint=True)
    elif choice == "5":
        train_raw(load_initial=False)
    elif choice == "6":
        train_raw(load_initial=True)
    elif choice == "7":
        train_raw(load_final=True)
    elif choice == "8":
        train_raw(resume_checkpoint=True)
    elif choice == "9":
        train_heatmap(load_initial=False)
    elif choice == "10":
        train_heatmap(load_initial=True)
    elif choice == "11":
        train_heatmap(load_final=True)
    elif choice == "12":
        train_heatmap(resume_checkpoint=True)
    else:
        print("Please select proper choice :3")

if __name__ == "__main__":
    main()
