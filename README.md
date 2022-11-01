## PAR: Political Actor Representation Learning with Social Context and Expert Knowledge @ EMNLP 2022.

paper link: [[link]](https://arxiv.org/abs/2210.08362)

### Political Actor Representations

You can first explore the learned representations of the 1,069 political actors, with names in `political_actor_reps/entity_list.txt` and representation vectors in `political_actor_reps/learned_reps.pt` using torch.load('learned_reps.pt') with size `[1069, 512]`.

### Political Actor Representation Learning Code

I uploaded a pre-processed version of the political actor representation learning code to `pretraining_src/`. `python PSP.py` would be enough, and the learned representations would be at `pretraining_src/representation.pt` when it finishes with size `[1069, 512]`.

The pre-processed network data is based on files in `pretraining_data/`, which I will briefly explain in the following:

- `entity2id.txt`: the name-to-id mapping of the 1071 entities in the network. Note that 1071 = 1069 + 2 since "liberal" and "conservative" are also added as two entities, which should be extracted out when PAR pre-training happens, since predicting legislators' stance towards liberal and conservative ideologies is one of the training objective of PAR.

- `entity_summary.txt`: the summary of the 1071 entities taken from Wikipedia.

- `relation2id.txt`: the relation-to-id mapping of the 10 relations in the network. Note that strongly_favor 5, favor 6, neutral 7, oppose 8, strongly_oppose 9 are discretized labels from the political think tanks, which should be treated as labels for the PAR training objective 1: expert knowledge.

- `triples.data`: the edge information of the network, with triples as (ei, rj, ek) representing the relation rj between entities ei and ek.

If you got confused about the above, just use the pre-processed data and code in `pretraining_src/`.

### Environment

`conda env create -f environment.yml` would generate a conda environment called `PAR` that should be able to run the code.

### To-do: Applying PAR Learned Representations to Downstream Tasks

I will be uploading these next week, hopefully!

In the meantime, should you have any questions or need me to upload a clean version of the rest of the code, please don't hesitate to reach out.

### Citation

```
@article{feng2022political,
  title={PAR: Political Actor Representation Learning with Social Context and Expert Knowledge},
  author={Feng, Shangbin and Tan, Zhaoxuan and Chen, Zilong and Wang, Ningnan and Yu, Peisheng and Zheng, Qinghua and Chang, Xiaojun and Luo, Minnan},
  journal={arXiv preprint arXiv:2210.08362},
  year={2022}
}
```