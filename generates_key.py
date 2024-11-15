import pickle
import pathlib as Path

import streamlit_authenticator as stauth

names = ["Nam", "Fidan", "Ojaswi", "Sterling", "Kim"]
usernames = ["hoang", "fidan", "ojashwi", "sterling", "kim"]
passwords= ["gannon123", "gannon123", "gannon123", "gannon123"]

hasher = stauth.Hasher(passwords)
# Hashing the passwords
hashed_passwords = hasher.generate_hashes()
file_path=Path(__file__).parent / "hashedPW.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)
