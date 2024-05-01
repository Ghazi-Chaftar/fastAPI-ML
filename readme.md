## Creating the Virtual Environment

1. Install virtualenv if you haven't already:

   ```bash
   pip install virtualenv
   ```

2. Create a virtual environment named `my_venv` (replace with your preferred name) using:

   ```bash
   virtualenv my_venv
   ```

## Activating the Virtual Environment

- **Windows**:

  ```bash
  my_venv\Scripts\activate.bat
  ```

## Installing Dependencies

1. Install dependencies from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Open server

Inside the activated virtual environment, run:

```bash
   uvicorn main:app --port 8000
```
