# What the agents need to know about elliptic2d

**The goal.** A neural operator (the student) is being pretrained on problems that you generate.
It sees two fields on a 32 × 32 grid, the log of the coefficient a and the source f, and predicts
the solution u. It will later be tested on problems it has never seen, such as Poisson problems,
smooth coefficients, high-contrast media and two-phase media. You will not be shown those tests.
Your job is to generate the problems the student learns most from right now.

**The equation.** Every problem is a steady diffusion problem on the unit square:

    −∇·(a(x, y) ∇u) = f(x, y)   on (0, 1)²,   u = 0 on the boundary

`a = a_lo + (a_hi − a_lo) · sigmoid(sharpness · g_a)`, where g_a is a random field with unit RMS
and spectral decay `a_slope` (larger means smoother). `sharpness` 0 gives a constant coefficient;
small values give a smooth one; large values give two distinct phases with sharp interfaces.
`f = f_mean + f_amp · g_f`, where g_f is another random field with decay `f_slope`. `seed` fixes
both random fields.

**How your generator is judged.** Each problem is solved with conjugate gradients at 64 × 64 and at
128 × 128. It is rejected if either solve fails to converge or if they disagree by more than 1%,
which is what happens when interfaces are too sharp to resolve. Rejected problems teach nothing.
A new version of the generator is measured with a short trial: the student trains on half of its
problems for one round, and the score is how much error that removed on the other half. Problems
the student has mastered teach nothing. Problems far harder than the generator's typical one are
out of reach: the student does not train on them yet. The training report shows, for each kind of
problem, how often that happened.

**What helps.** Coverage of contrasts from 1 to hundreds, of smooth and sharp media, and of
uniform and varying sources. Sharp media near, but not past, the resolution limit. Coefficient
fields at several length scales.

**Hard limits** (enforced by `check`; a problem outside them is dropped): a_lo [0.01, 10],
a_hi [0.01, 100] and at least a_lo, contrast a_hi / a_lo at most 1000; a_slope and f_slope
[0.5, 6]; sharpness [0, 50]; f_mean [−10, 10]; f_amp [0, 10], with f not zero everywhere;
seed an int in [0, 2³²).
