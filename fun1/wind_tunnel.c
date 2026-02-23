#include "raylib.h"
#include <math.h>
#include <stdio.h>

// air_density
// absolute air pressure / gas constant for air * absolute tempature in Kelvin

// wind pressure fomrula
// pressure = 1/2 * rho(air density) * v^2 (win speed in m/s or mps)


// wind load 
// wind pressure ^ * effective surface (total_surface * sin(angle))

// Soft constants
#define TEMPERATURE 20       // Celsius
#define WIND_SPEED 100       // m/s
#define HEIGHT_M 1000        // height in meters

// Hard constants
#define P0 101325.0          // Sea-level standard pressure in Pa
#define T0 288.15            // Sea-level standard temperature in K
#define L 0.0065             // Temperature lapse rate in K/m
#define g 9.80665            // Gravity in m/s^2
#define M 0.0289644          // Molar mass of air in kg/mol
#define R 8.31432            // Universal gas constant in J/(mol·K)

double air_pressure_at_altitude(double h) {
    if (h < 0) h = 0;
    double term = 1.0 - (L * h / T0);
    double exponent = (g * M) / (R * L);
    return P0 * pow(term, exponent);
}

double air_density(double air_pressure, double temperature_in_celsius) {
    // Convert Celsius to Kelvin
    double temperature_K = temperature_in_celsius + 273.15;
    return air_pressure / (R / M * temperature_K); // Use R/M for air-specific gas constant
}

double wind_pressure(double air_density_value, double wind_speed_value) {
    return 0.5 * air_density_value * pow(wind_speed_value, 2);
}

int main() {

  double new_speed = WIND_SPEED;

    for (int i = 0; i < 1000; i++){
    double pressure = air_pressure_at_altitude(HEIGHT_M);  // Pa
    double density = air_density(pressure, TEMPERATURE); // kg/m^3
    double wind_press = wind_pressure(density, WIND_SPEED+i); // Pa
    
    printf("Wind pressure at %i meter altitude with %i speed: %f Pa\n",HEIGHT_M, WIND_SPEED+i, wind_press);

  }

    return 0;
}
