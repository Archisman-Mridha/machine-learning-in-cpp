#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;

float traingingData[][2]= {
  {0, 0}, {1, 2}, {2, 4}, {3, 6}, {4, 8}
};

#define trainingDataRowCount (sizeof(traingingData) / sizeof(traingingData[0]))

float getRandomParameterValue( ) {
  return (float)rand( ) / (float)RAND_MAX;
}

float evaluateLoss(float w) {
  float squaredError= 0;
  for(size_t i= 0; i < trainingDataRowCount; i++) {
    float x= traingingData[i][0],
      y= x * w;

    squaredError += (traingingData[i][1] - y) * (traingingData[i][1] - y);
  }
  // Loss Function value
  float meanSquaredError= squaredError / trainingDataRowCount;

  return meanSquaredError;
}

int main( ) {
  // Set the seed different everytime,
  // so that the value of random number generated is different everytime.
  // srand(time(0));

  srand(69);

  // w is the parameter
  float w= getRandomParameterValue( ) * 10.0f;

  // Using the method of finite differences to adjust the value of w.
  // Our goal is to minimize the loss function and tend it towards 0.
  float epsilon= 1e-3,
    learningRate= 1e-3;
  for(size_t i= 0; i< 500; ++i) {
    // By derivate we mean dLoss/dw.
    float derivative= (evaluateLoss(w + epsilon) - evaluateLoss(w)) / epsilon;
    // For machine learning scenario, even the derivative is a large value.
    // Thats'y we introduce a factor called the learning rate.
    w -= learningRate * derivative;
  }

  // Check how the generated value of the parameter fits our data model
  float loss= evaluateLoss(w);
  cout<<"Mean Squared Error - "<<loss<<endl;

  cout<<"Estimated value of w - "<<w<<endl;

  return 0;
}