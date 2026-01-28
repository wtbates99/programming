//  gcc orbit.c -o orbit.o -lraylib && ./orbit.o
#include "raylib.h"
#include <stdlib.h>

int main(void) {
  const int screenWidth = 800;
  const int screenHeight = 450;

  InitWindow(screenWidth, screenHeight, "Orbit");

  // optional to limit performance
  SetTargetFPS(60);

  float timer = 0.0f;
  int currentCircle = 0;
  int rand_x = 0;
  int rand_y = 0;
  int rand_z = 0;

  while (!WindowShouldClose()) {
    timer += GetFrameTime();
    if (timer >= 0.1f) {
      timer = 0.0f;
      currentCircle++;
      rand_x = (rand() % (screenWidth - 1)) + 1;
      rand_y = (rand() % (screenHeight - 1)) + 1;
      rand_z = rand() % 50 + 10;
    }
    BeginDrawing();
    ClearBackground(RAYWHITE);
    DrawText(TextFormat("FPS: %i || RandX: %i RandY %i RandZ %i", GetFPS(),
                        rand_x, rand_y, rand_z),
             10, 10, 10, BLACK);
    DrawCircle(rand_x, rand_y, rand_z, RED);
    EndDrawing();
  }

  CloseWindow();
  return 0;
}
