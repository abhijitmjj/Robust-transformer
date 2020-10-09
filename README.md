<h1> Robust-transformer </h1>

Robust Transformer is an end-to-end functional style distributed data transformation pipeline built on top of Google JAX


<p align="left">
  <img src="transformer.png" width="500" title="hover text">
</p>

<h2> Pure Functions </h2>  
`split_shuffle(*, key: jax.numpy.lax_numpy.ndarray, raw_data: pd.DataFrame, n: int) -> Dict[str, pd.DataFrame]`  
`inner_split(*, key: jax.numpy.lax_numpy.ndarray, outer_fold_data: Dict[str, pd.DataFrame], n: int) -> Dict[str, pd.DataFrame]`

<h2> Composable Transform </h2>  

    toolz.compose(lambda x: inner_loop((yield from x)), outer_loop) (key=rng_input, raw_data=tester_df, n=5)   
