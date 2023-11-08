# for i in {1..99}
# do
#   dir_num=$(printf "%04d" $i)
#   python generate_json_file.py --input_folder=$dir_num
# done

for i in {90..99}
do
  dir_num=$(printf "%04d" $i)
  torchrun --standalone --nnodes=1 --nproc-per-node=4 inference.py --from_pretrained cogvlm-chat --version chat --english --fp16 --input_file=/f_ndata/zekai/test_json/$dir_num.jsonl --output_file=/f_ndata/zekai/caption_data/$dir_num.txt
done

# torchrun --standalone --nnodes=1 --nproc-per-node=8 inference.py --from_pretrained cogvlm-chat --version chat --english --fp16 --input_file=test_json/0001.jsonl --output_file=outputs/0001.txt