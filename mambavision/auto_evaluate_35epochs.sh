#!/bin/bash
# Auto-evaluate MambaVision when training completes

echo "Monitoring MambaVision training..."

# Wait for training to complete
while true; do
    if grep -q "Training complete" mamba_35epochs_training.log 2>/dev/null; then
        echo "✅ Training completed!"
        sleep 30  # Wait for model to be saved
        
        echo ""
        echo "=== Running Evaluation ==="
        
        # Evaluate accuracy
        python3 evaluate.py --checkpoint checkpoints/best.pth 2>&1 | tee mamba_35epochs_evaluation.log
        
        echo ""
        echo "=== Measuring Inference Speed ==="
        
        # Measure speed
        python3 measure_inference.py 2>&1 | tee mamba_35epochs_speed.log
        
        echo ""
        echo "✅ Evaluation complete!"
        echo "Results saved to:"
        echo "  - mamba_35epochs_evaluation.log"
        echo "  - mamba_35epochs_speed.log"
        
        break
    fi
    
    # Check progress every 2 minutes
    sleep 120
    echo "Still training... ($(date +%H:%M:%S))"
    grep "Epoch \[" mamba_35epochs_training.log 2>/dev/null | tail -1
done
