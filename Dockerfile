FROM python:3.12-slim-bullseye

# Set working directory
WORKDIR /app

# copy source code
COPY . .

# install dependencies with poetry
RUN pip install poetry==1.8.3
RUN poetry install --without development

# Expose port
EXPOSE 8501

# Run the app
CMD ["poetry", "run", "streamlit", "run", "streamlit_app.py"]