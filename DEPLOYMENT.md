# Deployment: Vercel + Render

The frontend is hosted on Vercel and the FastAPI inference service on Render.
The Render service reads the model weights and inference files from this repo.
The multi-gigabyte training arrays and raw weather data are excluded by
`.gitignore` and are not needed for prediction.

## Publish the repository

Create a GitHub repository, then from this project folder run:

```powershell
git init
git add .
git status --short
```

Confirm `data/processed/X.npy`, `data/processed/y.npy`, `data/processed/cloudburst_X.npy`,
`data/processed/cloudburst_y.npy`, and `data/raw/` are not staged. The trained model
weights and the small inference `.npy` files under `data/processed/` must be staged.
Then commit and push using the commands GitHub provides for the new repository.

## Deploy the API on Render

1. In Render, create a **Blueprint** from the GitHub repository. It will read
   `render.yaml` and create `weather-nowcasting-api`.
2. After the first deploy, copy the service URL, for example
   `https://weather-nowcasting-api.onrender.com`.
3. Set the Render environment variable `FRONTEND_ORIGINS` to the Vercel site
   origin (for example `https://your-project.vercel.app`, with no trailing slash).
   Save the change and redeploy. The API health check is at `/api/health`.

## Deploy the frontend on Vercel

1. Import the same GitHub repository as a Vercel project.
2. Set **Root Directory** to `frontend`.
3. Add the environment variable `VITE_API_BASE` with the Render service URL,
   without a trailing slash, then deploy.
4. Copy the deployed Vercel origin into Render's `FRONTEND_ORIGINS` as above.

The app calls the API from the browser, so both environment values must match
the deployed origins. If you use a custom domain, set that exact frontend origin
in `FRONTEND_ORIGINS` and redeploy the Render service.

## Runtime notes

- Render's free service can sleep while idle; the first request may be slow.
- Live predictions fetch multiple Open-Meteo batches and can take tens of seconds.
- The CSV prediction logs are stored on the service filesystem. On a free Render
  instance they are ephemeral and may be lost when the service restarts or deploys.
- If the service runs out of memory while importing TensorFlow or predicting,
  move it to a Render instance with more memory.