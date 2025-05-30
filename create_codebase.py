import os

file_paths = [
    "/Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/src/custom_cnn.py",
    "/Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/src/data_manager.py",
    "/Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/src/evalue_agent.py",
    "/Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/src/live_training_server.py",
    "/Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/src/trading_env.py",
    "/Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/src/train_agent.py",
    "/Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/src/visualization_callback.py",
    "/Users/melihkarakose/Desktop/EC 581/btc_rl/rl_trader/visualizer/live_training_visualizer.html"
]

output_path = "/Users/melihkarakose/Desktop/rl_codebase_snapshot.txt"

with open(output_path, "w", encoding="utf-8") as outfile:
    for path in file_paths:
        filename = os.path.basename(path)
        outfile.write(f"{filename}:\n\n")
        try:
            with open(path, "r", encoding="utf-8") as infile:
                content = infile.read()
                # Temizlik: gereksiz boşluklar normalize
                cleaned = content.strip().replace('\r\n', '\n')
                outfile.write(cleaned)
        except Exception as e:
            outfile.write(f"[!! ERROR reading file: {e}]\n")
        outfile.write("\n\n" + "#"*100 + "\n\n")

print(f"Yazıldı knk: {output_path} — Şimdi LLM'e rahatça verebilirsin 🚀")
