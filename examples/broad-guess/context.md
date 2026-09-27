# What the agents need to know about pde1d

**The goal.** A neural operator (the student) is being pretrained on problems that you generate.
It sees four frames of a 1D field, 0.05 time units apart, and predicts the next frame. It is
never told the PDE. It will later be tested on physics it has never seen, such as advection,
diffusion, viscous Burgers, KdV solitons and reaction–diffusion fronts. You will not be shown
those tests. Your job is to generate the problems the student learns most from right now.

**The equation.** Every problem is one PDE on the periodic interval [0, 2π):

    u_t = ν u_xx − μ u_xxxx − c u_x − δ u_xxx + r u − a u u_x + b u² + g u³ + φ F(x)

`Problem.coef` holds (ν, μ, c, δ, r, a, b, g, φ) in the order of `TERMS`. A zero turns a term
off. The initial field is a random band-limited field with RMS amplitude `ic_amp`, spectral
slope `ic_slope` (larger means smoother) and mean `ic_mean`. `ic_squash` maps it through a
sigmoid into (0, 1), which suits reaction fronts. `t0_frames` delays the first frame, so the
student also sees developed states. `seed` fixes the random fields.

**How your generator is judged.** Each problem is solved exactly on the GPU. A problem is
rejected if it blows up or if it is under-resolved (the 256- and 512-point solutions disagree).
Rejected problems teach nothing. A new version of the generator is measured with a short trial:
the student trains on half of its problems for one round, and the score is how much error that
removed on the other half. Problems the student has mastered teach nothing. Chaotic or noise-like
problems teach little: they are hard, but there is little to learn. Problems far harder than the
generator's typical one are out of reach: the student does not train on them yet, so they teach
nothing either. The training report shows, for each kind of problem, how often that happened.

**What helps.** Coverage of different kinds of dynamics, amplitudes and time scales. Problems
near, but not past, the edge of stability. Mixes of terms that create structure: fronts, shocks
that viscosity keeps resolved, dispersive wave trains, travelling waves.

**Hard limits** (enforced by `check`; a problem outside them is dropped):
diffusion [0, 1], hyperdiffusion [0, 0.01], advection [−5, 5], dispersion [−0.1, 0.1],
growth [−3, 3], burgers / quadratic / cubic / forcing [−5, 5], at least one term nonzero;
ic_amp [0.001, 5], ic_slope [0, 8], ic_mean [−3, 3]; t0_frames an int in [0, 10]; seed an int
in [0, 2³²).
