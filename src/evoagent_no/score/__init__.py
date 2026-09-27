"""The self-play score: how much the student is learning from each problem right now.

See `progress.learning_progress`: the learning-progress reward of Cowsik et al. (arXiv
2609.30063), computed for a whole batch in one forward-mode (JVP) pass, divided by the problem's
loss by default so that hard-but-unlearnable problems do not take over the curriculum.
"""
