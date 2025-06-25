# 1. Define Image Name and Tag
#    (Replace with your actual GitHub username and repository name)
GITHUB_USER="hw33ni"
REPO_NAME="stellatrain_cac"
IMAGE_TAG="main" # Or "latest", "v1.1", etc.
FULL_IMAGE_NAME="ghcr.io/${GITHUB_USER}/${REPO_NAME}:${IMAGE_TAG}"

# ghcr.io/hw33ni/stellatrain_cac:main
# 2. Build the Docker Image
#    (-t tags the image, . uses current directory as build context)
docker build -t "${FULL_IMAGE_NAME}" .

# 3. Log in to GitHub Container Registry (if not already logged in or session expired)
#    (You'll be prompted for your PAT as the password)
docker login ghcr.io -u "${GITHUB_USER}"

# 4. Push the Docker Image to GHCR
docker push "${FULL_IMAGE_NAME}"

# 5. (Optional) Verify on GitHub Packages page for your repository
echo "Build and push complete. Check https://github.com/${GITHUB_USER}/${REPO_NAME}/packages"

# docker run -p 5555:5555 -p 5556:5556 -p 70555:70555 -p 70556:70556 -it --rm --gpus all --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864
