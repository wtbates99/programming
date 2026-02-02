// gcc breathing_circles.c -o breathing_circles.o -lraylib -lm &&
// ./breathing_circles.o

#include "raylib.h"
#include <math.h>

#define MAX_CIRCLES 200
#define NEW_CIRCLES 10
#define MIN_LIFE 2.0f
#define MAX_LIFE 5.0f

typedef struct {
  float x, y;
  float radius;
  Color color;
  float age;      // how long it's been alive
  float lifeTime; // total lifespan
} Circle;

int main(void) {
  const int screenWidth = 1600;
  const int screenHeight = 900;

  InitWindow(screenWidth, screenHeight, "Breathing Circles");
  SetTargetFPS(60);

  Circle circles[MAX_CIRCLES];
  int circleCount = 0;

  while (!WindowShouldClose()) {
    float dt = GetFrameTime();

    // Spawn new circles if below max
    int spawnCount = (circleCount + NEW_CIRCLES > MAX_CIRCLES)
                         ? MAX_CIRCLES - circleCount
                         : NEW_CIRCLES;
    for (int i = 0; i < spawnCount; i++) {
      circles[circleCount].x = (float)GetRandomValue(50, screenWidth - 50);
      circles[circleCount].y = (float)GetRandomValue(50, screenHeight - 50);
      circles[circleCount].radius = (float)GetRandomValue(10, 100);
      circles[circleCount].color =
          (Color){(unsigned char)GetRandomValue(0, 255),
                  (unsigned char)GetRandomValue(0, 255),
                  (unsigned char)GetRandomValue(0, 255), 255};
      circles[circleCount].age = 0.0f;
      circles[circleCount].lifeTime =
          (float)GetRandomValue((int)(MIN_LIFE * 1000),
                                (int)(MAX_LIFE * 1000)) /
          1000.0f;
      circleCount++;
    }

    // Update circles
    for (int i = 0; i < circleCount; i++) {
      circles[i].age += dt;

      // Remove circle if life exceeded
      if (circles[i].age >= circles[i].lifeTime) {
        circles[i] = circles[circleCount - 1];
        circleCount--;
        i--;
        continue;
      }
    }

    // Draw
    BeginDrawing();
    ClearBackground(BLACK);
    for (int i = 0; i < circleCount; i++) {
      float alpha =
          sinf((circles[i].age / circles[i].lifeTime) * 3.14159f) * 255.0f;
      Color c = circles[i].color;
      c.a = (unsigned char)alpha;
      DrawCircle((int)circles[i].x, (int)circles[i].y, circles[i].radius, c);
    }
    DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, 20, BLACK);
    EndDrawing();
  }

  CloseWindow();
  return 0;
}
