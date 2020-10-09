<h1> Robust-transformer </h1>

Robust Transformer - a tool for designing reproducible Machine learning experiments.
It’s an end to end functional style data transformation distributed pipeline built on top of Google JAX.
The functional style and fine controlled training protocols will help users in getting clean reproducible results with minimum hassle. It’s inspired from Haskell, so most of the functions and annotations might seem familiar to Functional programming enthusiasts 
Feel free to collaborate and contribute if you’re interested.


<p align="left">
  <img src="transformer.png" width="500" title="hover text">
</p>

<h2> Pure Functions </h2>  

    split_shuffle(*, key: jax.numpy.lax_numpy.ndarray, raw_data: pd.DataFrame, n: int) -> Dict[str, pd.DataFrame] 
    
    inner_split(*, key: jax.numpy.lax_numpy.ndarray, outer_fold_data: Dict[str, pd.DataFrame], n: int) -> Dict[str, pd.DataFrame]

<h2> Composable Transform </h2>  

    toolz.compose(lambda x: inner_loop((yield from x)), outer_loop) (key=rng_input, raw_data=tester_df, n=5)   
