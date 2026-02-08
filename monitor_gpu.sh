while true; do
    echo "===== $(date) =====" >> gpu_log.txt
    nvidia-smi --query-gpu=index,utilization.gpu,power.draw,power.limit,temperature.gpu --format=csv >> gpu_log.txt
    echo "" >> gpu_log.txt
    sleep 0.25
done
