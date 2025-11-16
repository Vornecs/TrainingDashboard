"""
FastAPI Server for AI Training Platform
=======================================
Provides REST API and WebSocket endpoints for the web interface.
Handles training management, dataset operations, and model inference.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import threading
import torch
import json
from pathlib import Path
import os
import tempfile

from model import GPTModel
from dataset import DatasetManager, DEFAULT_SAMPLE_TEXT
from trainer import Trainer


# ==================== Pydantic Models ====================

class DatasetAddTextRequest(BaseModel):
    text: str
    source: Optional[str] = "manual"


class DatasetAddURLRequest(BaseModel):
    url: str


# File uploads are handled via UploadFile, no Pydantic model needed


class TrainConfigRequest(BaseModel):
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    seq_length: int = 128
    # Model architecture
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 1024


class ChatRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.8


class LoadModelRequest(BaseModel):
    checkpoint_path: str


# ==================== Application State ====================

class AppState:
    """Global application state"""

    def __init__(self):
        self.dataset_manager: Optional[DatasetManager] = None
        self.model: Optional[GPTModel] = None
        self.trainer: Optional[Trainer] = None
        self.training_thread: Optional[threading.Thread] = None
        self.websocket_clients: List[WebSocket] = []
        self.is_training = False
        self.training_config = None

    def reset_dataset(self):
        """Reset dataset manager"""
        self.dataset_manager = DatasetManager()


app_state = AppState()


# ==================== FastAPI App ====================

app = FastAPI(
    title="AI Training Platform",
    description="Train transformer models from scratch with real-time monitoring",
    version="1.0.0"
)

# CORS middleware (allow frontend to communicate)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== WebSocket Manager ====================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                dead_connections.append(connection)

        # Clean up dead connections
        for conn in dead_connections:
            if conn in self.active_connections:
                self.active_connections.remove(conn)


manager = ConnectionManager()


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "AI Training Platform API",
        "is_training": app_state.is_training,
        "has_dataset": app_state.dataset_manager is not None,
        "has_model": app_state.model is not None
    }


@app.post("/dataset/reset")
async def reset_dataset():
    """Reset the dataset"""
    if app_state.is_training:
        raise HTTPException(400, "Cannot reset dataset while training")

    app_state.reset_dataset()
    return {"message": "Dataset reset successfully"}


@app.post("/dataset/add-text")
async def add_text_to_dataset(request: DatasetAddTextRequest):
    """Add raw text to the dataset"""
    if app_state.dataset_manager is None:
        app_state.reset_dataset()

    app_state.dataset_manager.add_text(request.text, request.source)

    return {
        "message": "Text added successfully",
        "total_sources": len(app_state.dataset_manager.raw_texts),
        "total_chars": len(app_state.dataset_manager.combined_text) if app_state.dataset_manager.combined_text else 0
    }


@app.post("/dataset/add-url")
async def add_url_to_dataset(request: DatasetAddURLRequest):
    """Fetch and add text from a URL"""
    if app_state.dataset_manager is None:
        app_state.reset_dataset()

    success = app_state.dataset_manager.add_url(request.url)

    if not success:
        raise HTTPException(400, f"Failed to fetch URL: {request.url}")

    return {
        "message": "URL content added successfully",
        "total_sources": len(app_state.dataset_manager.raw_texts),
        "url": request.url
    }


@app.post("/dataset/add-file")
async def add_file_to_dataset(file: UploadFile = File(...)):
    """Add text from an uploaded file"""
    if app_state.dataset_manager is None:
        app_state.reset_dataset()

    # Validate file type (allow text files)
    allowed_extensions = ['.txt', '.md', '.csv', '.json', '.log', '.py', '.js', '.html', '.xml']
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}")

    try:
        # Read file content
        content = await file.read()

        # Decode as text
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            # Try other common encodings
            try:
                text = content.decode('latin-1')
            except:
                raise HTTPException(400, "Could not decode file. Please ensure it's a valid text file.")

        # Add to dataset
        app_state.dataset_manager.add_text(text, f"file:{file.filename}")

        return {
            "message": "File added successfully",
            "total_sources": len(app_state.dataset_manager.raw_texts),
            "filename": file.filename,
            "size_bytes": len(content),
            "chars": len(text)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to process file: {str(e)}")


@app.get("/dataset/info")
async def get_dataset_info():
    """Get information about the current dataset"""
    if app_state.dataset_manager is None:
        return {"has_data": False}

    return {
        "has_data": True,
        "num_sources": len(app_state.dataset_manager.raw_texts),
        "total_chars": len(app_state.dataset_manager.combined_text) if app_state.dataset_manager.combined_text else sum(
            item['length'] for item in app_state.dataset_manager.raw_texts),
        "sources": [{"source": item['source'], "length": item['length']} for item in
                    app_state.dataset_manager.raw_texts],
        "sample": app_state.dataset_manager.get_sample_text() if app_state.dataset_manager.raw_texts else ""
    }


@app.post("/dataset/add-sample")
async def add_sample_data():
    """Add default sample text for quick testing"""
    if app_state.dataset_manager is None:
        app_state.reset_dataset()

    app_state.dataset_manager.add_text(DEFAULT_SAMPLE_TEXT, "sample_data")

    return {
        "message": "Sample data added successfully",
        "total_sources": len(app_state.dataset_manager.raw_texts)
    }


@app.post("/training/start")
async def start_training(config: TrainConfigRequest):
    """Start training the model"""
    if app_state.is_training:
        raise HTTPException(400, "Training is already in progress")

    if app_state.dataset_manager is None or not app_state.dataset_manager.raw_texts:
        raise HTTPException(400, "No dataset available. Add data first.")

    # Store config
    app_state.training_config = config

    # Prepare dataset
    app_state.dataset_manager.seq_length = config.seq_length
    app_state.dataset_manager.prepare()

    # Create dataloader
    dataloader = app_state.dataset_manager.create_dataloader(
        batch_size=config.batch_size,
        shuffle=True
    )

    # Initialize model
    app_state.model = GPTModel(
        vocab_size=app_state.dataset_manager.tokenizer.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_seq_len=config.seq_length
    )

    # Initialize trainer
    app_state.trainer = Trainer(
        model=app_state.model,
        train_dataloader=dataloader,
        learning_rate=config.learning_rate,
        max_epochs=config.epochs,
        checkpoint_dir='./checkpoints'
    )

    # Add callbacks for real-time updates
    async def step_callback(step_info):
        await manager.broadcast({
            "type": "training_step",
            "data": step_info
        })

    async def epoch_callback(epoch_info):
        await manager.broadcast({
            "type": "training_epoch",
            "data": epoch_info
        })

    # Wrap async callbacks for sync trainer
    def sync_step_callback(step_info):
        asyncio.run(step_callback(step_info))

    def sync_epoch_callback(epoch_info):
        asyncio.run(epoch_callback(epoch_info))

    app_state.trainer.add_step_callback(sync_step_callback)
    app_state.trainer.add_epoch_callback(sync_epoch_callback)

    # Start training in background thread
    def train_background():
        app_state.is_training = True
        try:
            app_state.trainer.train()
        finally:
            app_state.is_training = False
            # Notify clients training is complete
            asyncio.run(manager.broadcast({
                "type": "training_complete",
                "data": {"message": "Training completed"}
            }))

    app_state.training_thread = threading.Thread(target=train_background)
    app_state.training_thread.start()

    return {
        "message": "Training started",
        "config": config.dict(),
        "model_parameters": app_state.model.get_num_parameters(),
        "vocab_size": app_state.dataset_manager.tokenizer.vocab_size
    }


@app.post("/training/stop")
async def stop_training():
    """Stop the training process"""
    if not app_state.is_training:
        raise HTTPException(400, "No training in progress")

    if app_state.trainer:
        app_state.trainer.stop_training()

    return {"message": "Training stop signal sent"}


@app.get("/training/status")
async def get_training_status():
    """Get current training status"""
    if not app_state.trainer:
        return {"is_training": False}

    return {
        "is_training": app_state.is_training,
        "current_epoch": app_state.trainer.current_epoch,
        "max_epochs": app_state.trainer.max_epochs,
        "global_step": app_state.trainer.global_step,
        "stats": app_state.trainer.training_stats
    }


@app.post("/chat")
async def chat_with_model(request: ChatRequest):
    """Generate text from the model"""
    if app_state.model is None:
        raise HTTPException(400, "No model available. Train a model first.")

    if app_state.dataset_manager is None or app_state.dataset_manager.tokenizer is None:
        raise HTTPException(400, "No tokenizer available")

    try:
        generated = app_state.trainer.generate_sample(
            prompt=request.prompt,
            tokenizer=app_state.dataset_manager.tokenizer,
            max_length=request.max_length,
            temperature=request.temperature
        )

        return {
            "prompt": request.prompt,
            "generated": generated,
            "length": len(generated)
        }
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")


@app.get("/checkpoints")
async def list_checkpoints():
    """List available model checkpoints"""
    checkpoint_dir = Path('./checkpoints')
    if not checkpoint_dir.exists():
        return {"checkpoints": []}

    checkpoints = [
        {
            "name": f.name,
            "path": str(f),
            "size_mb": f.stat().st_size / (1024 * 1024)
        }
        for f in checkpoint_dir.glob("*.pt")
    ]

    return {"checkpoints": checkpoints}


@app.post("/model/load")
async def load_model(request: LoadModelRequest):
    """Load a model from checkpoint"""
    if app_state.is_training:
        raise HTTPException(400, "Cannot load model while training")

    try:
        checkpoint = torch.load(request.checkpoint_path, map_location='cpu')

        # Initialize model with saved config
        config = checkpoint['model_config']
        app_state.model = GPTModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            max_seq_len=config['max_seq_len']
        )

        app_state.model.load_state_dict(checkpoint['model_state_dict'])

        return {
            "message": "Model loaded successfully",
            "checkpoint": request.checkpoint_path,
            "epoch": checkpoint.get('epoch', 'unknown')
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {str(e)}")


# ==================== WebSocket Endpoint ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time training updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and receive messages
            data = await websocket.receive_text()

            # Echo back for testing
            await websocket.send_json({
                "type": "echo",
                "data": data
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ==================== Startup ====================

@app.on_event("startup")
async def startup_event():
    """Initialize app state on startup"""
    print("="*50)
    print("AI Training Platform Server Starting")
    print("="*50)

    # Create directories
    Path('./checkpoints').mkdir(exist_ok=True)
    Path('./data').mkdir(exist_ok=True)

    print("Server ready!")
    print("Access the API at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("="*50)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
