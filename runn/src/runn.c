// Rami's Usable Neural Network
// Version: 0.0.0.1

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "runn.h"

/* Neural Network */

// ------------------------
// NN->Activation Functions
// ------------------------

// Linear
float ActivationLinear(float val)
{
	return val;
}

float ActivationLinearDeriv(float val)
{
	return 1.0;
}

// Sigmoid
float ActivationSigmoid(float val)
{
	return 1.0 / (1.0 + exp(-val));
}

float ActivationSigmoidDeriv(float val)
{
	float sigmoid = ActivationSigmoid(val);
	return sigmoid * (1.0 - sigmoid);
}

// Tanh
float ActivationTanh(float val)
{
	return tanh(val);
}

float ActivationTanhDeriv(float val)
{
	double _tanh = tanh(val);
	return 1.0 - _tanh * _tanh;
}

// ReLU
float ActivationReLU(float val)
{
	return (val >= 0.0 ? val : 0.0);
}

float ActivationReLUDeriv(float val)
{
	return (val >= 0.0 ? 1.0 : 0.0);
}

// ------------------
// NN->Loss Functions
// ------------------

float LossMSE(size_t size, float act[], float exp[])
{
	float sum = 0;
	for (size_t i = 0; i < size; i++)
		sum += pow(act[i]-exp[i], 2);
	return sum/size;
}

void LossMSEDeriv(size_t size, float act[], float exp[], float out[])
{
	for (size_t i = 0; i < size; i++)
		out[i] = 2.0*(act[i]-exp[i])/size;
}

// ---------
// NN->Layer
// ---------

bool NNLayerAlloc(NNLayer *layer, NNLayerParams params, size_t nsize)
{
	*layer = (NNLayer){
		.size       = params.size,
		.weights    = NULL,
		.biases     = NULL,
		.denseIn    = calloc(params.size, sizeof(*layer->denseIn)),
		.activIn    = calloc(nsize, sizeof(*layer->activIn)),
		.activation = params.activation };

	if (nsize != 0)
	{
		layer->weights = calloc(nsize*params.size, sizeof(*layer->weights));
		layer->biases  = calloc(nsize,             sizeof(*layer->biases));
		
		return
			   layer->weights
			&& layer->biases
			&& layer->denseIn
			&& layer->activIn;
	}
	
	return true;
}

void NNLayerFree(NNLayer *layer)
{
	free(layer->weights);
	free(layer->biases);
	free(layer->denseIn);
	free(layer->activIn);
}


// ---------------------
// NeuralNetwork (logic)
// ---------------------

// Allows to in == out
bool NNLayerForward(NeuralNetwork *nn, size_t lindex, float *in, float *out)
{
	NNLayer *layer = &nn->layers[lindex];
	size_t size = nn->layers[lindex].size;
	size_t nsize = nn->layers[lindex+1].size;
	// Setting dense layer X
	for (int r = 0; r < size; r++)
		layer->denseIn[r] = in[r];

	// Y = func(W dot X + B)
	for (int r = 0; r < nsize; r++)
	{
		// Setting activ layer X
		layer->activIn[r] = layer->biases[r];
		for (int c = 0; c < size; c++)
			layer->activIn[r] += layer->weights[r*size+c] * layer->denseIn[c];
		// Setting out
		out[r] = layer->activation.func(layer->activIn[r]);
	}
}

// Does NOT allows to in == out
bool NNLayerBackwardGD(
	NeuralNetwork *nn, size_t lindex, float gradOut[], float gradIn[], float lrate)
{	
	NNLayer *layer = &nn->layers[lindex];
	size_t size = nn->layers[lindex].size;
	size_t nsize = nn->layers[lindex+1].size;
	// Activation layer
	// Xa - activation layer in 
	// dE/dXa = dE/dY * f'(Xa)
	float *gradDenseOut = calloc(nsize, sizeof(*gradDenseOut));
	for (size_t i = 0; i < nsize; i++)
		gradDenseOut[i] = gradOut[i] * layer->activation.deriv(layer->activIn[i]);

	// Dense layer
	// X - dense layer in 
	// dE/dW = dE/dY * X^T 
	// W = W' - lrate * dE/dW
	// B = B' - lrate * dE/dW
	for (int i = 0; i < nsize; i++)
	{
		for (int j = 0; j < size; j++)
			layer->weights[i*size+j] -= lrate * gradDenseOut[i] * layer->denseIn[j];
		layer->biases[i] -= lrate * gradDenseOut[i];
	}

	// Return gradIn
	for (int i = 0; i < size; i++)
	{
		gradIn[i] = 0.0;
		for (int j = 0; j < nsize; j++)
			gradIn[i] += layer->weights[j*nsize+i] * gradDenseOut[j];
	}
	free(gradDenseOut);
}

bool NNAlloc(
	NeuralNetwork *nn,
	size_t lcount,
	NNLayerParams lparams[])
{
	*nn = (NeuralNetwork){
		.lcount = lcount,
		.layers = malloc(lcount*sizeof(*nn->layers))
	};

	if (!nn->layers)
		return false;

	for (size_t i = 0; i < nn->lcount; i++)
	{
		if (!NNLayerAlloc(
			nn->layers+i,
			lparams[i],
			(i != nn->lcount-1 ? lparams[i+1].size : 0)))
		{
			for (int j = 0; j < i; j++)
				NNLayerFree(nn->layers+i);

			free(nn->layers);

			return false;
		}
	}

	return true;
}

void NNFree(NeuralNetwork *nn)
{
	if (nn->layers == NULL)
		return;

	for (size_t i = 0; i < nn->lcount; i++)
		NNLayerFree(nn->layers+i);

	free(nn->layers);
}

void NNForward(NeuralNetwork *nn, float in[], float out[])
{
	size_t maxLayerSize = 0;
	for (size_t l = 0; l < nn->lcount; l++)
		if (nn->layers[l].size > maxLayerSize)
			maxLayerSize = nn->layers[l].size;

	// Create the overshoot matrix
	float *buffer = malloc(maxLayerSize*sizeof(*buffer));

	for (size_t i = 0; i < nn->layers[0].size; i++)
		buffer[i] = in[i];

	// Feed forward
	for (size_t l = 0; l < nn->lcount-1; l++)
		NNLayerForward(nn, l, buffer, buffer);

	for (size_t i = 0; i < nn->layers[nn->lcount-1].size; i++)
		out[i] = buffer[i];
	
	free(buffer);
}

void ArrayRandomize(float *arr, float from, float to, size_t size)
{
	for (size_t i = 0; i < size; i++)
		arr[i] = from+(float)rand()/RAND_MAX*(to-from);
}

void NNShuffle(NeuralNetwork *nn)
{	
	for (size_t l = 0; l < nn->lcount-1; l++)
	{
		ArrayRandomize(
			nn->layers[l].weights,
			-1.0,
			+1.0,
			nn->layers[l+1].size * nn->layers[l].size
		);

		ArrayRandomize(
			nn->layers[l].biases,
			-1.0,
			+1.0,
			nn->layers[l+1].size
		);
	}
}
