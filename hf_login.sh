#!/bin/bash

# Path to your token file
TOKEN_FILE="/pvc/hf_token.txt"
# Or use a hidden file or one in a config directory:
# TOKEN_FILE="$HOME/.config/hf_token"

# --- Check if token file exists ---
if [ ! -f "$TOKEN_FILE" ]; then
  echo "Error: Token file not found at '$TOKEN_FILE'"
  exit 1
fi

# --- Read the token from the file ---
# 'cat' reads the file, 'tr -d' removes newline characters just in case
HF_TOKEN=$(cat "$TOKEN_FILE" | tr -d '\n\r')

# --- Check if token was read successfully ---
if [ -z "$HF_TOKEN" ]; then
  echo "Error: Token file '$TOKEN_FILE' appears to be empty."
  exit 1
fi

# --- Perform login using the token ---
echo "Attempting Hugging Face login using token from '$TOKEN_FILE'..."
huggingface-cli login --token "$HF_TOKEN"

# --- Check the exit status of the login command ---
LOGIN_STATUS=$?
if [ $LOGIN_STATUS -eq 0 ]; then
  echo "Login successful (or token already cached by CLI)."
else
  echo "Login failed with status code $LOGIN_STATUS."
  exit 1
fi

echo "Proceeding with subsequent commands..."
# Add other commands here that require login
# e.g., git push ... (to a private HF repo)
# or running a script that uses cached credentials.