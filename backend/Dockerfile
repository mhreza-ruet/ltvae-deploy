# syntax=docker/dockerfile:1
FROM mambaorg/micromamba:1.5.8

ARG ENV_NAME=ltvae
ARG PY=3.12

# Create a conda env with RDKit + PyTorch (CPU) + FastAPI stack
RUN micromamba create -y -n $ENV_NAME -c conda-forge -c pytorch \
    python=${PY} \
    rdkit=2024.03.5 \
    pytorch=2.4.* cpuonly \
    uvicorn=0.30.* \
    fastapi=0.115.* \
    pydantic=2.9.* \
    pillow>=10.0 \
    boto3=1.35.* \
    numpy=1.26.* \
 && micromamba clean -a -y
SHELL ["bash", "-lc"]

# Activate env
ENV MAMBA_DEFAULT_ENV=${ENV_NAME}
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:$PATH

# Prevent OpenMP duplicate warning/crash
ENV KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# Copy your code
COPY ./app /app/app

# (If your checkpoints & vocab are already under app/, nothing else to copy)
# Otherwise later we’ll set env vars for S3 or different paths.

EXPOSE 8000

# Healthcheck hits your /health route
HEALTHCHECK CMD wget -qO- http://127.0.0.1:8000/health || exit 1

# Start API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]