#include "raylib.h"
#include <vector>
#include <math.h>

Mesh CreateF1CarMesh() {
    Mesh mesh = {0};
    mesh.vertices = NULL;
    mesh.normals = NULL;
    mesh.texcoords = NULL;
    mesh.texcoords2 = NULL;
    mesh.colors = NULL;
    mesh.tangents = NULL;
    mesh.indices = NULL;

    std::vector<Vector3> vertices;
    std::vector<unsigned short> indices;
    std::vector<Vector3> normals;

    // Body vertices
    vertices.push_back({-1.5f,0.0f,-4.0f});
    vertices.push_back({ 1.5f,0.0f,-4.0f});
    vertices.push_back({ 1.5f,0.0f, 4.0f});
    vertices.push_back({-1.5f,0.0f, 4.0f});
    vertices.push_back({-1.5f,0.5f,-4.0f});
    vertices.push_back({ 1.5f,0.5f,-4.0f});
    vertices.push_back({ 1.5f,0.5f, 4.0f});
    vertices.push_back({-1.5f,0.5f, 4.0f});

    int bodyFaces[36] = {
        0,1,2, 0,2,3,
        4,5,6, 4,6,7,
        0,1,5, 0,5,4,
        3,2,6, 3,6,7,
        1,2,6, 1,6,5,
        0,3,7, 0,7,4
    };

    for(int i=0;i<36;i++) indices.push_back(bodyFaces[i]);

    for(size_t i=0;i<vertices.size();i++) normals.push_back({0,1,0});

    // Allocate arrays
    mesh.vertexCount = vertices.size();
    mesh.triangleCount = indices.size()/3;

    mesh.vertices = (float *)RL_MALLOC(vertices.size()*3*sizeof(float));
    mesh.normals  = (float *)RL_MALLOC(normals.size()*3*sizeof(float));
    mesh.indices  = (unsigned short *)RL_MALLOC(indices.size()*sizeof(unsigned short));

    for(size_t i=0;i<vertices.size();i++){
        mesh.vertices[i*3+0] = vertices[i].x;
        mesh.vertices[i*3+1] = vertices[i].y;
        mesh.vertices[i*3+2] = vertices[i].z;
        mesh.normals[i*3+0] = normals[i].x;
        mesh.normals[i*3+1] = normals[i].y;
        mesh.normals[i*3+2] = normals[i].z;
    }
    for(size_t i=0;i<indices.size();i++) mesh.indices[i] = indices[i];

    return mesh;
}

int main() {
    InitWindow(1280,720,"Procedural F1 Car");

    Camera3D camera = {0};
    camera.position = (Vector3){8,5,8};
    camera.target   = (Vector3){0,0,0};
    camera.up       = (Vector3){0,1,0};
    camera.fovy     = 45.0f;

    Mesh carMesh = CreateF1CarMesh();
    Model carModel = LoadModelFromMesh(carMesh);

    SetTargetFPS(60);

    while(!WindowShouldClose()){
        // Manual camera movement
        if (IsKeyDown(KEY_W)) camera.position.z -= 0.1f;
        if (IsKeyDown(KEY_S)) camera.position.z += 0.1f;
        if (IsKeyDown(KEY_A)) camera.position.x -= 0.1f;
        if (IsKeyDown(KEY_D)) camera.position.x += 0.1f;
        if (IsKeyDown(KEY_Q)) camera.position.y += 0.1f;
        if (IsKeyDown(KEY_E)) camera.position.y -= 0.1f;

        BeginDrawing();
        ClearBackground(RAYWHITE);

        BeginMode3D(camera);
        DrawModel(carModel,(Vector3){0,0,0},1.0f,RED);
        DrawGrid(20,1.0f);
        EndMode3D();

        DrawText("Use WASDQE to move camera",10,10,20,DARKGRAY);
        EndDrawing();
    }

    UnloadModel(carModel);
    CloseWindow();
    return 0;
}
