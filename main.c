#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "raylib.h"
#include "runn.h"

#define BACKGROUND_COLOR (Color){32,25,40,255}
#define TEXT_COLOR (Color){200,160,250,120} 

#define INPUT_NEURON_COLOR (Color){200,200,220,255}
#define POSITIVE_COLOR (Color){50,200,64,255}
#define NEGATIVE_COLOR (Color){200,50,64,255}

#define NEURON_RADIUS 20

void BackwardNN(NeuralNetwork *nn, float out[], float eIn[], float eOut[], float gradOut[], float gradIn[], float *mse)
{
	NNForward(nn, eIn, out);
	LossMSEDeriv(3, out, eOut, gradOut);
	*mse = LossMSE(1, out, eOut);

	for (int l = nn->lcount-2; l >= 0; l--)
	{
		NNLayerBackwardGD(nn, l, gradOut, gradIn, (float)rand()/RAND_MAX*(*mse));
		for (int k = 0; k < 3; k++)
			gradOut[k] = gradIn[k];
	}
}

void DrawNNLayer(NeuralNetwork *nn, Rectangle rect, int l, float hGap, float vGap)
{
	float layerHeight = vGap * (nn->layers[l].size - 1);
	float nextLayerHeight = 0;
	int nextX = 0;
	if (l != nn->lcount-1)
	{
		nextLayerHeight = vGap * (nn->layers[l+1].size - 1);
		nextX = rect.x + hGap * (l + 1);
	}

	for (int n = 0; n < nn->layers[l].size; n++)
	{
		int currX = rect.x + hGap * l;
		int currY = rect.y + vGap * n + rect.height/2 - layerHeight/2;

		if (l != nn->lcount-1)
		{
			for (int N = 0; N < nn->layers[l+1].size; N++)
			{
				int nextY = rect.y + vGap * N + rect.height/2 - nextLayerHeight/2;
				float weight = nn->layers[l].weights[N*nn->layers[l+1].size+n];
				Color wcolor = (weight > 0 ? POSITIVE_COLOR : NEGATIVE_COLOR);
				wcolor.a = fmax(0.25, fmin(1.0, fabs(weight))) * 180;

				DrawLineEx(
					(Vector2){ currX + NEURON_RADIUS, currY },
					(Vector2){ nextX - NEURON_RADIUS, nextY },
					4,
					wcolor);
			}
		}
		
		Color ncolor = INPUT_NEURON_COLOR;

		if (l != 0)
		{
			ncolor = (nn->layers[l-1].biases[n] > 0.0 ? POSITIVE_COLOR : NEGATIVE_COLOR);
			ncolor.a = fmax(0.5, fmin(1.0, fabs(nn->layers[l-1].biases[n]))) * 255;
		}

		DrawCircle(currX, currY, NEURON_RADIUS, ncolor);
	}
}

void DrawNN(NeuralNetwork *nn, Rectangle rect)
{
	int maxLayerSize = 0;
	for (int l = 0; l < nn->lcount; l++)
		if (maxLayerSize < nn->layers[l].size)
			maxLayerSize = nn->layers[l].size;

	float hGap = rect.width / (nn->lcount - 1);
	float vGap = rect.height / (maxLayerSize - 1);

	//DrawRectangleLinesEx(rect, 2, TEXT_COLOR);

	for (int l = 0; l < nn->lcount; l++)
	{
		DrawNNLayer(nn, rect, l, hGap, vGap);
	}
}

int main()
{
	srand(time(NULL));

	const int screenWidth = 800;
	const int screenHeight = 600;

	SetConfigFlags(FLAG_MSAA_4X_HINT);

	InitWindow(screenWidth, screenHeight, "runn.c");
	
	SetTargetFPS(120);

	NeuralNetwork nn;

	NNLayerParams lparams[] = {
		{.size=2, .activation=ACTIVATION_TANH},
		{.size=3, .activation=ACTIVATION_TANH},
		{.size=1}
	};

	if (!NNAlloc(&nn, sizeof(lparams)/sizeof(*lparams), lparams))
		return 1;
	
	NNShuffle(&nn);

	float eIn[4][2]  = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
	float eOut[4][1] = { { 0    }, { 1    }, { 1    }, { 0    } };

	float out[1] = { 0 };
	float gradOut[3] = { 0, 0, 0 };
	float gradIn[3] = { 0, 0, 0 };

	int dindex = 0;

	bool stopTraining = true, start = false;
	
	float mse = 0.0;

	while (!WindowShouldClose()) {
		
		BeginDrawing();
			
			ClearBackground(BACKGROUND_COLOR);
	
			DrawNN(&nn, (Rectangle){ 200, 100, 400, 400 });

			if (IsKeyPressed(KEY_SPACE))
			{
				stopTraining = !stopTraining;
				start = true;
			}

			DrawText((stopTraining ? "NOT TRAINING" : "TRAINING"), 10, 10, 20, TEXT_COLOR);

			if (!stopTraining)
			{
				dindex = rand()%4;
				BackwardNN(&nn, out, eIn[dindex], eOut[dindex], gradOut, gradIn, &mse);
			}

			if (start)
			{
				DrawText(TextFormat("IN 1: %02.05f", eIn[dindex][0]), 10, 50, 20, TEXT_COLOR);
				DrawText(TextFormat("IN 2: %02.05f", eIn[dindex][1]), 10, 70, 20, TEXT_COLOR);
				DrawText(TextFormat("OUT: %02.05f", out[0]), 10, 90, 20, TEXT_COLOR);
				DrawText(TextFormat("MSE: %02.05f", mse), 10, 30, 20, TEXT_COLOR);
			}

		EndDrawing(); 
	}

	NNFree(&nn);

	CloseWindow();

	return 0;
}
