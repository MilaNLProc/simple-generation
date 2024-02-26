from simple_generation.speech import SimpleTranscriber
from datasets import load_dataset
import torch


model = "openai/whisper-large-v3"
transcriber = SimpleTranscriber(
    model_name_or_path=model,
    tgt_lang="es",
    torch_dtype=torch.bfloat16,
    device="cuda:0",
)


data = load_dataset(
    "google/fleurs", "es_419", split="test", trust_remote_code=True, streaming=True
)

samples = [r["audio"]["array"] for r in data.take(16)]
transcriptions = transcriber(samples, sampling_rate=16000, batch_size=2)
print(transcriptions)
