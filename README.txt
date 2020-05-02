——————EXPERIMENT_1————————
model is composed by encoder, K-means label, classifier
	K-means label: using mini-batch K-means clustering to labelize the data
	encoder: generalise the low_dimention representation of original data
	classifier: using the generalized data to match the K-means clustering results, to ensure the encoded vector
		maintain precise representation

the idea is mainly inspired by <Deep Clustering for Unsupervised Learning of Visual Features - Mathilde et. al.>