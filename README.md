MedAL Improvements

Work in progress.


### Roadmap:
- reproduce MedAL ourselves to get a baseline.
  - plain inception baseline in PyTorch
  - plain MedAL baseline in PyTorch

- Test the min max algorithm and get results.
- Test the min max + feature tranform on the hypersphere (soln 1 below)
- Try knn min-max approach on hypersphere (soln 2 below)
- Try the knn graph over time with momentum (soln 3 below)
  - plot features of a set of points (using t-sne or something similar) every al iteration

- Online learning: Don't retrain from scratch at every AL iteration.
  a) Try sampling N points and training incrementally on just those N
  b) Try sampling N points and train incrementally on those N plus a
    random subset of previously trained on points
  c) Try sampling N points and train incrementally on the whole dataset
  --> Repeat these 3 experiments using the KNN graph over time with momentum.

- If knn works, can try variations on soln (4).

Done:
x bridges pipeline and infrastructure
x redo the data processing steps to work with corrected Messidor.



---

Alex's ramblings... to be better organized

- There exists a min max approach proposed by Pedro and implemented by
  Anuj / Bique to switch from MedAL's averaging method on the feature
  vectors to select an unlabeled point to a min-max method.  Pdf
  description attached.

### Issues with min max approach:
  - The idea is to find unlabeled points that are the most unlike the current set of training points.
    Using the current min max method, we find an unlabeled feature vector with max distance from closest train feature
    vector.  Without loss of generalization, we can assume the vector is 2D.
    Suppose we visualize the points (from both labeled and unlabeled sets) in 2D space, identify a center point in the
    middle of the points and draw a circle with radius sufficiently large to capture most of the points.

    If we assume the points from each set are uniformly or normally distributed and
    equally likely to be closest, most of the time, the closest train point will
    be inside the circle.  This means that the furthest unlabeled point will rarely be in
    the middle of the circle.

    I suspect that this will lead to an uneven covering of the space, with
    concentration in the tails of the distribution (ie usually choose points
    near the boundary of the circle).  In other words, unlabelled points that
    happen to lie in the center of the circle will rarely or never get chosen.

  - A second issue is that if we always choose the unlabeled item with max
    distance to the chosen train point, we may sometimes pick an unlabeled item
    that is very close to an item already in the training set.

### Possible solutions:
  1 If we could constrain/transform/train all feature vectors to lie uniformly
  on the hypersphere (ie the perimeter of the 2D circle or surface of a 3D
  sphere or ...).  In this case, we never have points in the center of the
  circle, so we solve the first issue.
      --> convert a vector f to hypersphere via g(f): f / ||f||_2

  2 If we again transform the feature vectors as above, we could apply another
    kind of min-max using k neighbors: we pick the unlabeled point whose k
    closest (train set) neighbors are farthest away.  I think this ensures we
    get a more even covering of the space and solves both issues.
    - details: We have a bipartite graph of labeled and unlabeled points.  For
      each unlabeled point, find k closest labeled points and draw edges where
      the edge weight is distance between the (labeled, unlabeled) points.
      Assign a score to each unlabeled point as the aggregation (ie sum) of
      edge weights.  Since this is a graph, we could get more
      sophisticated with how to do aggregation but prob not worth it yet...

  3 Other possible extension/idea:  KNN Graph over time.  It might be possible
    that the feature vector from previous iterations contains information lost
    in the current iteration.  Since the KNN graph changes from iteration to
    iteration, we could probably keep some sort of history and include decay
    weighted edges from previous iterations.
    - the problem presents itself differently if we retrain from scratch vs
      train incrementally.
  4 Other possible extension/idea: KNN graph in the other direction.  We
    defined a bipartite graph of labeled and unlabeled points.  For each
    labeled point, find k closest unlabeled points.  From the resulting KNN
    graph, query unlabeled points with smallest degree.  We could increase k
    until there is only a small set of unlabeled nodes with smallest degree.
    - I wonder if this works better if the feature space isn't actually
      Euclidean.
