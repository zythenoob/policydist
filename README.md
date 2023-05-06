# Surprise Policy Distillation

Course project CS566.

Team members Jiaye Zhu, Rongcan Fang, Runqi Pang, and Divam Divam Kesharwani

Check the [final report](https://drive.google.com/file/d/1Jte2YH7-8V-idU1yPmJfRVYXesNC2uzM/view?usp=share_link)

## Dependencies

- pytorch
- [Ablator](https://github.com/fostiropoulos/ablator)
- gymnasium
- ray

## Run the code

To run a single experiment

`python main.py --config configs/spd.yaml`

To run distributed experiments

`ray start --head`

`python distributed.py --config configs/distributed/walker_spd.yaml`
 
