#include <iostream>
#include <cmath>
#include <iomanip>


int main() {
    double a, b, c;
    std::cin >> a >> b >> c;
    
    if (a == 0 && b == 0 && c != 0) {
        std::cout << "incorrect\n";
    } else if (a == 0 && b == 0 && c == 0) {
        std::cout << "any\n";
    } else if (a == 0) {
        std::cout << std::fixed << std::setprecision(6) << -c / b << '\n';
    } else {
        double discriminant = b * b - 4 * a * c;

        if (discriminant > 0) {
            double x1 = (-b + std::sqrt(discriminant)) / (2 * a);
            double x2 = (-b - std::sqrt(discriminant)) / (2 * a);

            std::cout << std::fixed << std::setprecision(6) << x1 << ' ' << x2 << '\n';
        } else if (discriminant == 0) {
            double x = -b / (2 * a);

            std::cout << std::fixed << std::setprecision(6) << x << '\n';
        } else {
            std::cout << "imaginary\n";
        }
    }

    return 0;
}