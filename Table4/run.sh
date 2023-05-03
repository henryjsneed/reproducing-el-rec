# ======================= Kaggle =====================
python facebook_dlrm_train_and_test.py --dataset=kaggle
sleep 15
python FAE_train_and_test.py --dataset=kaggle
sleep 15
python TTRec_train_and_test.py --dataset=kaggle
sleep 15

python ELRec_train_and_test.py --dataset=kaggle

# ======================= Avazu =====================
python facebook_dlrm_train_and_test.py --dataset=avazu
sleep 15
python FAE_train_and_test.py --dataset=avazu
sleep 15
python TTRec_train_and_test.py --dataset=avazu
sleep 15
python ELRec_train_and_test.py --dataset=avazu
sleep 15

# ==================== terabyte =====================
python facebook_dlrm_train_and_test.py --dataset=terabyte
sleep 15
python FAE_train_and_test.py --dataset=terabyte
sleep 15
python TTRec_train_and_test.py --dataset=terabyte
sleep 15
python ELRec_train_and_test.py --dataset=terabyte
