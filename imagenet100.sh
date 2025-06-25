#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Define directories and files
INITIAL_WORKING_DIR=$(pwd) # Save the initial directory
DATA_PARENT_DIR="/home"
DATA_DIR_NAME="data"
DATA_DIR="${DATA_PARENT_DIR}/${DATA_DIR_NAME}" # /home/data

ZIP_FILE_NAME="imagenet100.zip"
ZIP_FILE_PATH="${DATA_DIR}/${ZIP_FILE_NAME}"
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/ambityga/imagenet100"

TARGET_PROJECT_DIR="/home/stellatrain/explore-dp" # Final directory to cd into

# 1. Ensure the data directory exists
echo "Ensuring data directory ${DATA_DIR} exists..."
mkdir -p "${DATA_DIR}"

# --- Operations within DATA_DIR ---
pushd "${DATA_DIR}" > /dev/null # Temporarily change to DATA_DIR, silence pushd output
echo "Changed directory to $(pwd)"

# 2. Download the dataset if it doesn't already exist
if [ ! -f "${ZIP_FILE_NAME}" ]; then # Check for file, not full path here since we cd'd
  echo "Downloading ${ZIP_FILE_NAME} from ${DATASET_URL}..."
  # KAGGLE DOWNLOAD (IMPORTANT: Requires Kaggle API setup or direct download link)
  # Option 1: Using Kaggle CLI (recommended if installed and configured)
  # if command -v kaggle &> /dev/null; then
  #   echo "Using Kaggle CLI to download..."
  #   kaggle datasets download -d ambityga/imagenet100 -f "${ZIP_FILE_NAME}" -p "." --quiet # Download to current dir (.)
  # else
  #   echo "Kaggle CLI not found, attempting curl (may require manual auth steps)..."
  #   curl -L -o "${ZIP_FILE_NAME}" "${DATASET_URL}"
  # fi

  # Using the provided curl command (ensure auth is handled if needed)
  curl -L -o "${ZIP_FILE_NAME}" "${DATASET_URL}"

  if [ $? -ne 0 ] || [ ! -f "${ZIP_FILE_NAME}" ]; then # Check curl success and file existence
    echo "Error: Download failed. Please check the URL and your Kaggle API authentication/setup."
    popd > /dev/null # Return to original directory before exiting
    exit 1
  fi
  echo "Download complete."
else
  echo "${ZIP_FILE_NAME} already exists in $(pwd). Skipping download."
fi

# 3. Unzip the dataset if it hasn't been unzipped effectively
EXPECTED_DIR_AFTER_UNZIP="train.X1" # Or another key directory/file from the zip

if [ ! -d "${EXPECTED_DIR_AFTER_UNZIP}" ]; then
  if [ -f "${ZIP_FILE_NAME}" ]; then
    echo "Unzipping ${ZIP_FILE_NAME}..."
    unzip -qq "${ZIP_FILE_NAME}" # -qq for quiet
    if [ $? -ne 0 ]; then
      echo "Error: Unzip failed."
      popd > /dev/null; exit 1
    fi
    echo "Unzip complete."
  else
    echo "Error: ${ZIP_FILE_NAME} not found, cannot unzip."
    popd > /dev/null; exit 1
  fi
else
  echo "Dataset appears to be already unzipped (found ${EXPECTED_DIR_AFTER_UNZIP} in $(pwd)). Skipping unzip."
fi


# 4. Create the target 'train' directory (if it's not already there from a previous run)
TARGET_TRAIN_DIR="train"
echo "Ensuring target train directory ${TARGET_TRAIN_DIR} exists..."
mkdir -p "${TARGET_TRAIN_DIR}"

# 5. Move contents from train.X1 through train.X4 into 'train' and remove source dirs
echo "Organizing train directories..."
for i in {1..4}; do
  SOURCE_TRAIN_SUBDIR="train.X${i}"
  if [ -d "${SOURCE_TRAIN_SUBDIR}" ]; then
    echo "Moving contents from ${SOURCE_TRAIN_SUBDIR} to ${TARGET_TRAIN_DIR}/ and removing ${SOURCE_TRAIN_SUBDIR}"
    # Using rsync to move contents and then remove the source directory
    # The trailing slash on source is important for rsync to copy contents
    if rsync -a --info=progress2 "${SOURCE_TRAIN_SUBDIR}/" "${TARGET_TRAIN_DIR}/"; then
        rm -rf "${SOURCE_TRAIN_SUBDIR}" # Remove the source directory after successful rsync
        echo "Successfully moved and removed ${SOURCE_TRAIN_SUBDIR}"
    else
        echo "Error: rsync failed for ${SOURCE_TRAIN_SUBDIR}. Skipping removal."
    fi
    # Alternative with mv (less safe if filenames conflict, more manual cleanup)
    # mv -v "${SOURCE_TRAIN_SUBDIR}"/* "${TARGET_TRAIN_DIR}/"
    # if [ $? -eq 0 ]; then
    #   rm -rf "${SOURCE_TRAIN_SUBDIR}" # Careful with rm -rf!
    #   echo "Successfully moved and removed ${SOURCE_TRAIN_SUBDIR}"
    # else
    #    echo "Error moving from ${SOURCE_TRAIN_SUBDIR}. Skipping removal."
    # fi
  else
    echo "Warning: Directory ${SOURCE_TRAIN_SUBDIR} not found in $(pwd)."
  fi
done

# 6. Rename 'val.X' to 'val'
SOURCE_VAL_DIR="val.X"
TARGET_VAL_DIR="val"
echo "Renaming validation directory..."
if [ -e "${SOURCE_VAL_DIR}" ]; then # Use -e to check for file or directory
  echo "Renaming ${SOURCE_VAL_DIR} to ${TARGET_VAL_DIR}"
  mv -v "${SOURCE_VAL_DIR}" "${TARGET_VAL_DIR}"
else
  echo "Warning: ${SOURCE_VAL_DIR} not found in $(pwd)."
fi

echo "Dataset preparation finished in $(pwd)."
popd > /dev/null # Return from DATA_DIR to the directory where the script was started

# 7. Move (cd) to the final target project directory
echo "Changing directory to ${TARGET_PROJECT_DIR}..."
mkdir -p "${TARGET_PROJECT_DIR}" # Ensure it exists
cd "${TARGET_PROJECT_DIR}"
echo "Script finished. Current directory: $(pwd)"