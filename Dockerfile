FROM python:3.11

# Set the working directory
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy the poetry.toml file
COPY poetry.toml .

# Install the Python dependencies
RUN poetry install --no-root

# Copy the rest of the application code
COPY . .

# Set the entrypoint
ENTRYPOINT ["poetry", "run", "python", "-m", "bnet"]