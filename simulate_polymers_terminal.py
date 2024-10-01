import argparse 
from functions import * 

# writing json is better
if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Simulate polymers using polychrom module")
    
    # Add arguments with keywords 
    parser.add_argument('--num_monomers', type=int, required=True, help="Number of monomers in each polymer chain")
    parser.add_argument('--num_polymers', type=int, required=True, help="Number of polymers within a PBC")
    parser.add_argument('--num_observations', type=int, required=True, help="Number of polymer conformations")
    parser.add_argument('--save_folder', type=str, required=True, help="Directory to the save folder")
    
    parser.add_argument('--monomer_types', type=int, required=False, nargs='+', help="Enter the types of monomers")
