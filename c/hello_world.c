//  gcc hello_world.c -o hello_raylib.o -lraylib && ./hello_raylib
//
#include "raylib.h"

int main(void) {
  const int screenWidth = 800;
  const int screenHeight = 450;

  InitWindow(screenWidth, screenHeight, "raylib hello world");

  SetTargetFPS(60);

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(RAYWHITE);

    DrawText("Hello, Raylib!", 190, 200, 40, DARKGRAY);

    EndDrawing();
  }

  CloseWindow();
  return 0;
}
