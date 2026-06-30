import hashlib
import json
import os
import csv
from getpass import getpass

def hash_password(user_input, salt=None):
    """
    Hashes the user's password input using PBKDF2 and a salt.
    If 'salt' is not provided, loads it from credentials file.
    """
    if salt is None:
        with open('.auth/credentials.json', 'r') as f:
            credentials = json.load(f)
        salt = credentials['salt']

    user_input_bytes = user_input.encode('utf-8')
    salt_bytes = salt.encode('utf-8')
    hashed_password = hashlib.pbkdf2_hmac('sha256', user_input_bytes, salt_bytes, 100_000)
    return hashed_password.hex()

def verify_password(user_input):
    """
    Compares a user-provided password attempt to the stored hash.
    """
    with open('.auth/credentials.json', 'r') as f:
        credentials = json.load(f)
    password = credentials['password']
    salt = credentials['salt']
    hashed_user_input = hash_password(user_input, salt)
    return hashed_user_input == password

# Based on NIST 2025 recommendations, see https://proton.me/blog/nist-password-guidelines
# For now: 8-64 characters, no further restrictions + blocklist of easy passwords
def check_password_complexity(password):
    with open('assets/common_passwords.csv', 'r') as f:
        reader = csv.reader(f)
        blocklist = [row[0] for row in reader]
    if len(password) < 8 or len(password) > 64:
        return False
    if password in blocklist:
        return False
    return True

def generate_salt(length=32):
    import secrets
    return secrets.token_hex(length)

def handle_initial_setup(credentials_path):
    """
    Handles the initial setup—set the first password if credentials are not yet set.
    """
    print("👋 Initial setup: Please set a new password for your account.\n")
    # Generate a secure new salt for first-time setup
    salt = generate_salt()
    while True:
        new_password = getpass("Set NEW password: ")
        print()  # blank line
        if not check_password_complexity(new_password):
            print("❌  The password does not meet the requirements. Please try again.\n")
            continue
        confirm_password = getpass("Re-enter NEW password: ")   
        print()
        if new_password != confirm_password:
            print("❌  The passwords do not match. Please try again.\n")
            continue
        # Passed all checks, set password and salt in credentials file
        hashed_pwd = hash_password(new_password, salt)
        with open(credentials_path, 'w') as f:
            json.dump({'password': hashed_pwd, 'salt': salt}, f, indent=4)
        print("\n" + "="*50)
        print("🎉 Password set successfully! Initial setup complete.")
        print("="*50 + "\n")
        break

def handle_password_change(credentials_path):
    """
    Handles the password change workflow (for existing users).
    """
    # Main loop for password change
    while True:
        print("🔑 Please verify your identity to continue.\n")
        current_password = getpass("Current password: ")
        print()  # blank line

        if verify_password(current_password):
            print("✅ Password verified successfully!\n")
            print("Now, let's set your new password.\n")
            print("-"*50)
            print("Password requirements:")
            print("  • 8-64 characters")
            print("  • Must NOT be a commonly used password")
            print("-"*50 + "\n")

            # Prompt for new password until the user succeeds
            with open(credentials_path, 'r') as f:
                credentials = json.load(f)
            salt = credentials['salt']
            while True:
                new_password = getpass("Enter NEW password: ")
                print()  # blank line

                if not check_password_complexity(new_password):
                    print("❌  The new password does not meet the requirements. Please try again.\n")
                    continue

                confirm_password = getpass("Re-enter NEW password: ")
                print()  # blank line

                if new_password != confirm_password:
                    print("❌  The passwords do not match. Please try again.\n")
                    continue

                # Passed all checks, update password using the same salt
                credentials['password'] = hash_password(new_password, salt)
                with open(credentials_path, 'w') as f:
                    json.dump(credentials, f, indent=4)

                print("\n" + "="*50)
                print("🎉 Password changed successfully!")
                print("="*50 + "\n")
                break  # Exit password change loop
            break  # Exit main loop after successful password change
        else:
            print("❌  Incorrect password. Please try again.\n")

if __name__ == "__main__":
    # CLI: Enhanced UI and spacing for password management

    print("\n" + "="*50)
    print("         Welcome to the Password Management System!")
    print("="*50 + "\n")

    credentials_path = '.auth/credentials.json'

    # Initial credentials file setup if it doesn't exist, or needs initialization
    if not os.path.exists(credentials_path):
        os.makedirs('.auth', exist_ok=True)
        # We'll handle salt + hash assignment in initial setup function
        with open(credentials_path, 'w') as f:
            json.dump({'password': '', 'salt': ''}, f, indent=4)
        print("Credentials file created successfully.")
        handle_initial_setup(credentials_path)
    else:
        # If credentials exist, but password field is empty, treat as initial setup
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
        if not credentials.get('password') or not credentials.get('salt'):
            print("Detected uninitialized password. Beginning initial setup.")
            handle_initial_setup(credentials_path)
        else:
            handle_password_change(credentials_path)