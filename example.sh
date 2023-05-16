CUDA_VISIBLE_DEVICES=0 python main_NMP_QEM_projection_cat_MLP.py --cuda --do_train --do_test --data_path ../data/NELL-betae -n 128 -b 512 -d 400 -g 20  -lr 0.00005 --max_steps 450001 --cpu_num 1 --valid_steps 15000 --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up.2in.3in.inp.pin.pni" --print_on_screen

CUDA_VISIBLE_DEVICES=0 python main_NMP_QEM_projection_cat_MLP.py --cuda --do_train --do_test --data_path ../data/FB15k-237-betae -n 128 -b 512 -d 400 -g 20  -lr 0.00005 --max_steps 450001 --cpu_num 1 --valid_steps 15000 --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up.2in.3in.inp.pin.pni" --print_on_screen

CUDA_VISIBLE_DEVICES=0 python main_NMP_QEM_projection_cat_MLP.py --cuda --do_train --do_test --data_path ../data/FB15k-betae -n 128 -b 512 -d 400 -g 20  -lr 0.00005 --max_steps 450001 --cpu_num 1 --valid_steps 15000 --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up.2in.3in.inp.pin.pni" --print_on_screen
