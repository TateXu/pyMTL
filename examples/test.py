import pickle
from pymtl.linear_regression import MTLRegression as mtl
trained = mtl(max_prior_iter=1000, prior_conv_tol=0.0001, C=1, C_style='ML')



with open('/home/jxu/File/Code/Git/temp_data.pkl', 'rb') as f:
	source_vec, y = pickle.load(f) 

X_pool = np.asarray(source_vec[:7])
y_pool = np.asarray(y[:7])

X_pool_ = X_pool.reshape(-1, X_pool.shape[-1])
y_pool_ = y_pool.reshape(-1)

# trained.fit(X_pool_, y_pool_)
trained.init_model(X_pool_.shape, y_pool_.shape)
trained.fit_multi_task(source_vec[:7], y[:7], verbose=False, n_jobs=1)

fit_multi_task(source_vec[:7], y[:7], verbose=False, _jobs=1)
