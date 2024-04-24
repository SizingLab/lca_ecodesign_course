import os

def set_brightway_path(directory: str):
    """
    Sets the brightway projects directory. 
    https://docs.brightway.dev/en/latest/content/faq/data_management.html
    """

    # Create directory to store Brightway project and databases
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Set project data directory with environment variable
    os.environ["BRIGHTWAY2_DIR"] = directory

    # Save environment variable for future use in a .env file
    env_file = ".env"
    with open(env_file, 'w+') as f: 
        f.write(f"BRIGHTWAY2_DIR={directory}\n")

    return os.getenv("BRIGHTWAY2_DIR")