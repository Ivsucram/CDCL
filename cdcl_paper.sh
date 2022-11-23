for SEED in 0 1 2 3 4 # Select your SEED
do
    python train.py --config 'configs/datasets/usps_mnist.yml' --experiment 'usps_mnist' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/mnist_usps.yml' --experiment 'mnist_usps' --seed=$SEED --device-id=0 --log-wandb

    python train.py --config 'configs/datasets/visda.yml' --experiment 'visda' --seed=$SEED --device-id=0 --log-wandb

    python train.py --config 'configs/datasets/office31_amazon_dslr.yml' --experiment 'office31_amazon_dslr' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/office31_amazon_webcam.yml' --experiment 'office31_amazon_webcam' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/office31_dslr_amazon.yml' --experiment 'office31_dslr_amazon' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/office31_dslr_webcam.yml' --experiment 'office31_dslr_webcam' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/office31_webcam_amazon.yml' --experiment 'office31_webcam_amazon' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/office31_webcam_dslr.yml' --experiment 'office31_webcam_dslr' --seed=$SEED --device-id=0 --log-wandb

    python train.py --config 'configs/datasets/officeHome_art_clipart.yml' --experiment 'officeHome_art_clipart' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/officeHome_art_product.yml' --experiment 'officeHome_art_product' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/officeHome_art_real.yml' --experiment 'officeHome_art_real' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/officeHome_clipart_art.yml' --experiment 'officeHome_clipart_art' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/officeHome_clipart_product.yml' --experiment 'officeHome_clipart_product' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/officeHome_clipart_real.yml' --experiment 'officeHome_clipart_real' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/officeHome_product_clipart.yml' --experiment 'officeHome_product_clipart' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/officeHome_product_art.yml' --experiment 'officeHome_product_art' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/officeHome_product_real.yml' --experiment 'officeHome_product_real' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/officeHome_real_art.yml' --experiment 'officeHome_real_art' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/officeHome_real_clipart.yml' --experiment 'officeHome_real_clipart' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/officeHome_real_product.yml' --experiment 'officeHome_real_product' --seed=$SEED --device-id=0 --log-wandb

    python train.py --config 'configs/datasets/domainnet_clipart_infograph.yml' --experiment 'domainnet_clipart_infograph' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_clipart_painting.yml' --experiment 'domainnet_clipart_painting' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_clipart_quickdraw.yml' --experiment 'domainnet_clipart_quickdraw' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_clipart_real.yml' --experiment 'domainnet_clipart_real' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_clipart_sketch.yml' --experiment 'domainnet_clipart_sketch' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_infograph_clipart.yml' --experiment 'domainnet_infograph_clipart' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_infograph_painting.yml' --experiment 'domainnet_infograph_painting' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_infograph_quickdraw.yml' --experiment 'domainnet_infograph_quickdraw' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_infograph_real.yml' --experiment 'domainnet_infograph_real' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_infograph_sketch.yml' --experiment 'domainnet_infograph_sketch' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_painting_clipart.yml' --experiment 'domainnet_painting_clipart' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_painting_infograph.yml' --experiment 'domainnet_painting_infograph' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_painting_quickdraw.yml' --experiment 'domainnet_painting_quickdraw' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_painting_real.yml' --experiment 'domainnet_painting_real' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_painting_sketch.yml' --experiment 'domainnet_painting_sketch' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_quickdraw_clipart.yml' --experiment 'domainnet_quickdraw_clipart' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_quickdraw_infograph.yml' --experiment 'domainnet_quickdraw_infograph' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_quickdraw_painting.yml' --experiment 'domainnet_quickdraw_painting' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_quickdraw_real.yml' --experiment 'domainnet_quickdraw_real' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_quickdraw_sketch.yml' --experiment 'domainnet_quickdraw_sketch' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_real_clipart.yml' --experiment 'domainnet_real_clipart' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_real_infograph.yml' --experiment 'domainnet_real_infograph' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_real_painting.yml' --experiment 'domainnet_real_painting' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_real_quickdraw.yml' --experiment 'domainnet_real_quickdraw' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_real_sketch.yml' --experiment 'domainnet_real_sketch' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_sketch_clipart.yml' --experiment 'domainnet_sketch_clipart' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_sketch_infograph.yml' --experiment 'domainnet_sketch_infograph' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_sketch_painting.yml' --experiment 'domainnet_sketch_painting' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_sketch_quickdraw.yml' --experiment 'domainnet_sketch_quickdraw' --seed=$SEED --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/domainnet_sketch_real.yml' --experiment 'domainnet_sketch_real' --seed=$SEED --device-id=0 --log-wandb

    python train.py --config 'configs/datasets/usps_mnist.yml' --experiment 'usps_mnist_ablation_A' --seed=$SEED --alpha-1=0 --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/mnist_usps.yml' --experiment 'mnist_usps_ablation_A' --seed=$SEED  --alpha-1=0 --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/visda.yml' --experiment 'visda_ablation_A' --seed=$SEED --alpha-1=0 --device-id=0 --log-wandb

    python train.py --config 'configs/datasets/usps_mnist.yml' --experiment 'usps_mnist_ablation_B' --seed=$SEED --alpha-2=0 --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/mnist_usps.yml' --experiment 'mnist_usps_ablation_B' --seed=$SEED  --alpha-2=0 --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/visda.yml' --experiment 'visda_ablation_B' --seed=$SEED --alpha-2=0 --device-id=0 --log-wandb

    python train.py --config 'configs/datasets/usps_mnist.yml' --experiment 'usps_mnist_ablation_C' --seed=$SEED --alpha-3=0 --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/mnist_usps.yml' --experiment 'mnist_usps_ablation_C' --seed=$SEED  --alpha-3=0 --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/visda.yml' --experiment 'visda_ablation_C' --seed=$SEED --alpha-3=0 --device-id=0 --log-wandb

    python train.py --config 'configs/datasets/usps_mnist.yml' --experiment 'usps_mnist_ablation_D' --seed=$SEED --alpha-5=0 --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/mnist_usps.yml' --experiment 'mnist_usps_ablation_D' --seed=$SEED  --alpha-5=0 --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/visda.yml' --experiment 'visda_ablation_D' --seed=$SEED --alpha-5=0 --device-id=0 --log-wandb

    python train.py --config 'configs/datasets/usps_mnist.yml' --experiment 'usps_mnist_ablation_E' --seed=$SEED --alpha-6=0 --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/mnist_usps.yml' --experiment 'mnist_usps_ablation_E' --seed=$SEED  --alpha-6=0 --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/visda.yml' --experiment 'visda_ablation_E' --seed=$SEED --alpha-6=0 --device-id=0 --log-wandb

    python train.py --config 'configs/datasets/usps_mnist.yml' --experiment 'usps_mnist_ablation_F' --seed=$SEED --alpha-1=0 --alpha-2=0 --alpha-3=0 --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/mnist_usps.yml' --experiment 'mnist_usps_ablation_F' --seed=$SEED  --alpha-1=0 --alpha-2=0 --alpha-3=0 --device-id=0 --log-wandb
    python train.py --config 'configs/datasets/visda.yml' --experiment 'visda_ablation_F' --seed=$SEED --alpha-1=0 --alpha-2=0 --alpha-3=0 --device-id=0 --log-wandb
done