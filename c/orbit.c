// gcc orbit.c -o orbit.o -lraylib -lm && ./orbit.o

#include "math.h"
#include "raylib.h"
typedef struct {
  float x, y;
  float radius;
  Color color;
  float orbit_angle;
  float orbit_radius;
  float orbit_speed;
} Planet;

int main(void) {
  const int screenWidth = 1600;
  const int screenHeight = 900;

  InitWindow(screenWidth, screenHeight, "Orbit");
  SetTargetFPS(60);
  Planet center_planet = {
      .x = screenWidth / 2.0f,
      .y = screenHeight / 2.0f,
      .radius = 80.0f,
      .color = RAYWHITE,
      .orbit_angle = 0.0f,
      .orbit_radius = 200.0f,
      .orbit_speed = 1.0f,
  };

  Planet rotating_planet = {
      .x = screenWidth / 1.5f,
      .y = screenHeight / 1.5f,
      .radius = 40.0f,
      .color = GREEN,
      .orbit_angle = 0.0f,
      .orbit_radius = 200.0f,
      .orbit_speed = 1.0f,

  };

  while (!WindowShouldClose()) {
    float dt = GetFrameTime();

    rotating_planet.orbit_angle += rotating_planet.orbit_speed * dt;

    rotating_planet.x = center_planet.x + cosf(rotating_planet.orbit_angle) *
                                              rotating_planet.orbit_radius;

    rotating_planet.y = center_planet.y + sinf(rotating_planet.orbit_angle) *
                                              rotating_planet.orbit_radius;

    BeginDrawing();
    ClearBackground(BLACK);

    DrawCircleV((Vector2){center_planet.x, center_planet.y},
                center_planet.radius, center_planet.color);

    DrawCircleV((Vector2){rotating_planet.x, rotating_planet.y},
                rotating_planet.radius, rotating_planet.color);

    DrawLine(center_planet.x, center_planet.y, rotating_planet.x,
             rotating_planet.y, RED);

    DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, 20, RAYWHITE);
    EndDrawing();
  }
  return 0;
}
