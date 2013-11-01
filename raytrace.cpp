/*
Copyright (c) 2013 Yucheng Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cv.h>
#include <highgui.h>

using namespace std;

double sqr(double x) { return x * x; }

struct Vector {
	double x, y, z;

	Vector(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
	Vector operator * (double b) const { return Vector(x * b, y * b, z * b); }
	Vector operator / (double b) const { return Vector(x / b, y / b, z / b); }
	Vector operator - (const Vector& b) const { return Vector(x - b.x, y - b.y, z - b.z); }
	Vector operator + (const Vector& b) const { return Vector(x + b.x, y + b.y, z + b.z); }
	Vector operator * (const Vector& b) const { return Vector(x * b.x, y * b.y, z * b.z); }
	Vector operator - () const { return Vector(-x, -y, -z); }

	static Vector inf() { return Vector(1e10, 1e10, 1e10); }
};

Vector operator*(double a, const Vector& b) { return Vector(a * b.x, a * b.y, a * b.z); }

typedef Vector Color;

double maxabs(const Color& x) { return max(abs(x.x), max(abs(x.y), abs(x.z))); }
Color clamp(const Color& x) { return Color(min(x.x, 1.), min(x.y, 1.), min(x.z, 1.)); }

double dot(const Vector& a, const Vector& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
Vector cross(const Vector& a, const Vector& b) {
	return Vector(a.y * b.z - a.z * b.y,
				  -a.x * b.z + a.z * b.x,
				  a.x * b.y - a.y * b.x);
}
double norm(const Vector& a) {
	return sqrt(dot(a, a));
}
Vector normalize(const Vector& a) {
	return a / norm(a);
}
Vector ortho(const Vector& a) {
	if (fabs(a.x) + fabs(a.y) > 1e-10) return normalize(Vector(a.y, -a.x, 0));
	else return normalize(Vector(1, 1, 0));
}

Vector reflect(const Vector& light, const Vector& normal) {
	return 2 * dot(light, normal) * normal - light;
}
Vector refract(const Vector& light, const Vector& normal, double eta) {
	double cos1 = dot(light, normal);

	double X = 1 - (1 - sqr(cos1)) / sqr(eta);
	if (X < 0) return Vector::inf();
	double cos2 = sqrt(X);

	return - light / eta - (cos2 - cos1 / eta) * normal;
}

struct Surface { // one-sided surface
	virtual Vector normal(Vector pos) const = 0;
	virtual Vector intersect(Vector pos, Vector dir) const = 0;

	Color sr, dr, st;
	double srn, stn;
	double ir;

	virtual Color SR(Vector pos) const { return sr; }     // Specular Reflection
	virtual Color DR(Vector pos) const { return dr; }     // Diffusion Reflection
	virtual Color ST(Vector pos) const { return st; }     // Specular Transmission

	virtual double SRn(Vector pos) const { return srn; }   // n of Specular Reflection
	virtual double STn(Vector pos) const { return stn; }   // n of Specular Transmission

	virtual double IR(Vector pos) const { return ir; }     // Index of Refraction
};

struct outerCylinder : Surface {
	Vector center, dir;
	double H, R;

	Vector normal(Vector x) const {
		Vector x2 = x - center;
		Vector x3 = x2 - dot(x2, dir) * dir;
		return normalize(x3);
	}

	Vector intersect(Vector x, Vector y) const {
		double y1 = dot(y, dir);
		Vector y2 = y - y1 * dir;

		Vector x0 = x - center;
		double x1 = dot(x0, dir);
		Vector x2 = x0 - x1 * dir;

		double A = dot(y2, y2);
		double B = 2 * dot(x2, y2);
		double C = dot(x2, x2) - sqr(R);

		if (fabs(A) < 1e-10) return Vector::inf();

		double Delta = sqr(B) - 4 * A * C;
		if (Delta < 1e-10) return Vector::inf();

		double k = (-B - sqrt(Delta)) / (2 * A);
		if (k < 1e-10) return Vector::inf();

		double i1 = x1 + k * y1;
		if (i1 < -H / 2 + 1e-10 || i1 > H / 2 - 1e-10) return Vector::inf();

		return x + y * k;
	}

	Color srh, drh, sth;
	double l;

	Color texture(Vector x, Color normal, Color highlight) const {
		double x1 = dot(x - center, dir) + H / 2;
		x1 = fabs(fmod(x1, l * 2) - l) / l;

		return highlight * x1 + normal * (1 - x1);
	}

//	Color SR(Vector x) const { return texture(x, sr, srh); }
//	Color DR(Vector x) const { return texture(x, dr, drh); }
//	Color ST(Vector x) const { return texture(x, st, sth); }

	outerCylinder(Vector center0, Vector dir0, double H0, double R0,
			Color sr0, Color dr0, Color st0,
			Color sr1, Color dr1, Color st1,
			double l0,
			double srn0, double stn0,
			double ir0) {
		center = center0;
		dir = dir0;
		H = H0; R = R0;
		
		dr = dr0; sr = sr0; st = st0;
		drh = dr1; srh = sr1; sth = st1;
		l = l0;
		srn = srn0; stn = stn0;
		ir = ir0;
	}
};

struct innerCylinder : Surface {
	Vector center, dir;
	double H, R;

	Vector normal(Vector x) const {
		Vector x2 = x - center;
		Vector x3 = x2 - dot(x2, dir) * dir;
		if (norm(x3) < 1e-10) printf("!!\n");
		return -normalize(x3);
	}

	Vector intersect(Vector x, Vector y) const {
		double y1 = dot(y, dir);
		Vector y2 = y - y1 * dir;

		Vector x0 = x - center;
		double x1 = dot(x0, dir);
		Vector x2 = x0 - x1 * dir;

		double A = dot(y2, y2);
		double B = 2 * dot(x2, y2);
		double C = dot(x2, x2) - sqr(R);

		if (fabs(A) < 1e-10) return Vector::inf();

		double Delta = sqr(B) - 4 * A * C;
		if (Delta < 1e-10) return Vector::inf();

		double k2 = (-B - sqrt(Delta)) / (2 * A);
		if (k2 > 1e-10) return Vector::inf();

		double k = (-B + sqrt(Delta)) / (2 * A);
		if (k < 1e-10) return Vector::inf();

		double i1 = x1 + k * y1;
		if (i1 < -H / 2 + 1e-10 || i1 > H / 2 - 1e-10) return Vector::inf();

		return x + y * k;
	}

	Color srh, drh, sth;
	double l;

	Color texture(Vector x, Color normal, Color highlight) const {
		double x1 = dot(x - center, dir) + H / 2;
		x1 = fabs(fmod(x1, l * 2) - l) / l;

		return highlight * x1 + normal * (1 - x1);
	}

	Color SR(Vector x) const { return texture(x, sr, srh); }
	Color DR(Vector x) const { return texture(x, dr, drh); }
	Color ST(Vector x) const { return texture(x, st, sth); }

	innerCylinder(Vector center0, Vector dir0, double H0, double R0,
			Color sr0, Color dr0, Color st0,
			Color sr1, Color dr1, Color st1,
			double l0,
			double srn0, double stn0,
			double ir0) {
		center = center0;
		dir = dir0;
		H = H0; R = R0;
		
		dr = dr0; sr = sr0; st = st0;
		drh = dr1; srh = sr1; sth = st1;
		l = l0;
		srn = srn0; stn = stn0;
		ir = ir0;
	}
};

struct Plate : Surface {
	Vector center, dir;
	double R;

	Vector normal(Vector x) const {
		return dir;
	}

	Vector intersect(Vector x, Vector y) const {
		Vector x0 = x - center;
		double x1 = dot(x0, dir);
		if (x1 < 1e-10) return Vector::inf();

		double y1 = dot(y, dir);
		if (y1 > -1e-10) return Vector::inf();

		double k = -x1 / y1;
		Vector x2 = x + k * y;
		if (norm(x2 - center) >= R + 1e-10) return Vector::inf();

		return x2;
	}

	Plate(Vector center0, Vector dir0, double R0,
			Color sr0, Color dr0, Color st0,
			double srn0, double stn0,
			double ir0) {
		center = center0;
		dir = dir0;
		R = R0;
		
		dr = dr0; sr = sr0; st = st0;
		srn = srn0; stn = stn0;
		ir = ir0;
	}
};

struct Scene {
	vector< pair<Vector, Color> > lightSource; // lightSource[0] is global diffusion
	vector<Surface*> surfaces;

	bool testIntersect(Vector pos1, Vector pos2) {
		for (int i = 0; i < surfaces.size(); i++) {
			if (norm(surfaces[i]->intersect(pos1, pos2 - pos1) - pos1) <
					norm(pos2 - pos1) - 1e-10) {

				return false;
			}
		}
		return true;
	}

	Color fromLightSource(Vector pos, const Surface* s, Vector eye) {
		Color ans = s->DR(pos) * lightSource[0].second;
		Vector normal = s->normal(pos);

		for (int i = 1; i < lightSource.size(); i++) {
			if (!testIntersect(lightSource[i].first, pos)) continue;

			Vector light = normalize(lightSource[i].first - pos);
			if (dot(light, normal) < 0) {
				Vector refraction = refract(light, -normal, 1 / s->IR(pos));
				if (norm(refraction) < 1e9 && dot(refraction, eye) > 0) {
					ans = ans + lightSource[i].second * pow(dot(refraction, eye), s->STn(pos)) * s->ST(pos);
				}
			} else {
					assert(dot(light, normal) > 0);
				ans = ans + lightSource[i].second * dot(light, normal) * s->DR(pos);

				Vector reflection = reflect(light, normal);
				if (dot(reflection, eye) > 0) {
					ans = ans + lightSource[i].second * pow(dot(reflection, eye), s->SRn(pos)) * s->SR(pos);
				}
			}
		}

		return ans;
	}

	Color ray(Vector pos, Vector dir, Color weight) {
		if (maxabs(weight) < 1e-3) return Color(0, 0, 0);

		Vector intersect = Vector::inf();
		int intersectSurface = -1;
		for (int i = 0; i < surfaces.size(); i++) {
			Vector x = surfaces[i]->intersect(pos, dir);
			if (norm(x) > 1e9) continue;

			if (norm(x - pos) < norm(intersect - pos)) {
				intersect = x;
				intersectSurface = i;
			}
		}

		if (intersectSurface == -1) {
			return Color(0, 0, 0);
		}

		Vector eye = normalize(pos - intersect);
		Color ans = weight * fromLightSource(intersect, surfaces[intersectSurface], eye);

		Vector normal = surfaces[intersectSurface]->normal(intersect);
		Vector reflection = reflect(eye, normal);
		ans = ans + ray(intersect, reflection,
				weight * surfaces[intersectSurface]->SR(intersect));

		double IR = surfaces[intersectSurface]->IR(intersect);
		Vector refraction = refract(eye, normal, IR);
		if (norm(refraction) < 1e9) {
			ans = ans + ray(intersect, refraction,
					weight * surfaces[intersectSurface]->ST(intersect));
		}

		return ans;
	}

	void cylinder(Vector center0, Vector dir0, double H0, double R0,
			Color sr0, Color dr0, Color st0,
			double srn0, double stn0,
			double ir0) {
		surfaces.push_back(new outerCylinder(center0, dir0, H0, R0, sr0, dr0, st0, sr0, dr0, st0, 1, srn0, stn0, ir0));
		surfaces.push_back(new innerCylinder(center0, dir0, H0, R0, sr0, dr0, st0, sr0, dr0, st0, 1, srn0, stn0, 1/ir0));

		Vector center = center0 + dir0 * H0 / 2;
		surfaces.push_back(new Plate(center,  dir0, R0, sr0, dr0, st0, srn0, stn0, ir0));
		surfaces.push_back(new Plate(center, -dir0, R0, sr0, dr0, st0, srn0, stn0, 1/ir0));

		center = center0 - dir0 * H0 / 2;
		surfaces.push_back(new Plate(center, -dir0, R0, sr0, dr0, st0, srn0, stn0, ir0));
		surfaces.push_back(new Plate(center,  dir0, R0, sr0, dr0, st0, srn0, stn0, 1/ir0));
	}

	void fullbox() {
		Color sr = Color(0.2, 0.2, 0.2);
		Color dr = Color(0.2, 0.2, 0.2);
		Color st = Color(0, 0, 0);
		double srn = 1, stn = 10, ir = 2;

		surfaces.push_back(new Plate(Vector(0, 0, -1), Vector(0, 0, 1), 4, sr, dr, st, srn, stn, ir));
		surfaces.push_back(new Plate(Vector(0, 0, 1), Vector(0, 0, -1), 4, sr, dr, st, srn, stn, ir));
		surfaces.push_back(new Plate(Vector(1, 0, 0), Vector(-1, 0, 0), 4, sr, dr, st, srn, stn, ir));
		surfaces.push_back(new Plate(Vector(-1, 0, 0), Vector(1, 0, 0), 4, sr, dr, st, srn, stn, ir));
		surfaces.push_back(new Plate(Vector(0, 1, 0), Vector(0, -1, 0), 4, sr, dr, st, srn, stn, ir));
		surfaces.push_back(new Plate(Vector(0, -1, 0), Vector(0, 1, 0), 4, sr, dr, st, srn, stn, ir));
	}

	void box() {
		Color sr1 = Color(0, 0, 0);
		Color dr1 = Color(0.6, 0.6, 0.6);
		Color sr2 = Color(0.5, 0.5, 0.5);
		Color dr2 = Color(0.2, 0.2, 0.2);

		Color st = Color(0, 0, 0);
		double srn = 1, stn = 10, ir = 2;

		surfaces.push_back(new Plate(Vector(0, 0, -1), Vector(0, 0, 1), 1e10, sr1, dr1, st, srn, stn, ir));
		surfaces.push_back(new Plate(Vector(-10, 0, 0), Vector(1, 0, 0), 1e10, sr1, dr1, st, srn, stn, ir));
		surfaces.push_back(new Plate(Vector(0, 0, 10), Vector(0, 0, -1), 1e10, sr1, dr1, st, srn, stn, ir));
		surfaces.push_back(new Plate(Vector(0, 10, 0), Vector(0, -1, 0), 1e10, sr1, dr1, st, srn, stn, ir));
		surfaces.push_back(new Plate(Vector(0, -10, 0), Vector(0, 1, 0), 1e10, sr1, dr1, st, srn, stn, ir));
		surfaces.push_back(new Plate(Vector(1, 0, 0), Vector(-1, 0, 0), 1e10, sr2, dr2, st, srn, stn, ir));
	}

	void light() {
		lightSource.push_back(make_pair(Vector(), Color(0.8, 0.8, 0.8)));
		lightSource.push_back(make_pair(Vector(-1, 5, 5), Color(1, 1, 1)));
		//lightSource.push_back(make_pair(Vector(-0.9, -0.9, 0.9), Color(0.5, 0.5, 0.5)));
//		lightSource.push_back(make_pair(Vector(0, 0, 0), Color(1, 1, 1)));
	}

	void branch(Vector center, Vector dir, double H, double R) {
		Color c  = Color(0.9, 0.7, 1);
		Color c2 = Color(1, 1, 1);

		Color sr = Color(1, 1, 1) * 0.1;
		Color dr  = c  * 0.1;
		Color dr2 = c2 * 0.1;
		Color st  = c  * 0.8;
		Color st2 = c2 * 0.8;
		double srn = 1, stn = 10, ir = 1.02;
		ir =  2;

		double l = 0.1;

		surfaces.push_back(new outerCylinder(center, dir, H, R, sr, dr, st, sr, dr2, st2, l, srn, stn, ir));
		surfaces.push_back(new innerCylinder(center, dir, H, R, sr, dr, st, sr, dr2, st2, l, srn, stn, 1/ir));

		surfaces.push_back(new Plate(center + dir * H / 2,  dir, R, sr, dr, st, srn, stn, ir));
		surfaces.push_back(new Plate(center + dir * H / 2, -dir, R, sr, dr, st, srn, stn, 1/ir));
	}

	void crosses() {
		for (int i = -4; i <= 4; i++)
			branch(Vector(i * 0.2, 0, 0), Vector(0, 1, 0), 1.5, 0.05);
		for (int i = -3; i <= 3; i++)
			branch(Vector(0, i * 0.2, 0.2), Vector(1, 0, 0), 1.5, 0.05);
	}

	void tree(Vector bot, Vector dir, double h, double r, int l) {
		branch(bot + dir * h / 2, dir, h, r);
		if (l >= 4) return;

		Vector dir2 = ortho(dir);
		Vector dir3 = cross(dir, dir2);

		double theta = double(rand()) / RAND_MAX * M_PI * 2;
//		printf("%lf\n", theta);
		
		for (int i = 1; i <= 2; i++) {
			double angle = theta + (i == 1 ? 0 : M_PI * 0.9);
			Vector dir4 = cos(angle) * dir2 + sin(angle) * dir3;
			Vector dir5 = normalize((1.3) * dir + dir4);

			tree(bot + dir * h * ((sqrt(5) - 1) / 2),
				 dir5,
				 h * ((sqrt(5) - 1) / 2),
				 r * ((sqrt(5) - 1) / 2),
				 l + i);
		}
	}

	void tree() {
		srand(12341234);
		tree(Vector(1, 0, -0.5), normalize(Vector(-1, 0, 0)), 1.5, 0.08, 0);
	}
} scene;

struct Viewport {
	Vector eye;
	Vector upperleft, right, bottom;

	int pictureRow;
	int pictureCol;

	static const int Sample = 4;
	static const double Theta = 0.1;

	Vector sample(int n) {
		int r = n / Sample;
		int c = n % Sample;

		double x = 1. / (Sample + 1) * (r - (Sample + 1) / 2.);
		double y = 1. / (Sample + 1) * (c - (Sample + 1) / 2.);

		static const double SinTheta = sin(Theta);
		static const double CosTheta = cos(Theta);

		double x2 =  x * CosTheta + y * SinTheta;
		double y2 = -x * SinTheta + y * CosTheta;

		return Vector(x2 + 0.5, y2 + 0.5, 0);
	}

	Color sample(int r, int c, int n) {
		Vector rand = sample(n);

		Vector pos = upperleft +
			(bottom - upperleft) * (r + rand.x) / pictureRow +
			(right - upperleft) * (c + rand.y) / pictureCol;

		return scene.ray(pos, pos - eye, Color(1, 1, 1));
	}

	Color pixel(int r, int c) {
		Color s1 = sample(r, c, 2);
		Color s2 = sample(r, c, 8);
		Color s3 = sample(r, c, 15);

		const double eps = 0.01;
		const int nSample = Sample * Sample;

		Color avg = (s1 + s2 + s3) / 3;
		if (maxabs(s1 - avg) < eps &&
			maxabs(s2 - avg) < eps &&
			maxabs(s3 - avg) < eps) {
			return avg;
		} else {
			Color sum = s1 + s2 + s3;
			for (int i = 0; i < nSample; i++) {
				if (i == 2 || i == 8 || i == 15) continue;
				sum = sum + sample(r, c, i);
			}
			return sum / nSample;
		}
	}
} viewport;

int main() {
	const int Row = 700, Col = 700;
	
	double theta = M_PI / 2 + 0.2;
	double a = 0.6;
	Vector view = Vector(cos(theta), 0, sin(theta)) * 0.5 + Vector(-0.7, 0, 0);
	Vector tangent = -Vector(-sin(theta), 0, cos(theta));

	viewport.eye       = view * 2;
	view = view + Vector(-0.1, -0.1, 0);
	viewport.upperleft = view - tangent * a - Vector(0, 1, 0) * a;
	viewport.right     = view - tangent * a + Vector(0, 1, 0) * a;
	viewport.bottom    = view + tangent * a - Vector(0, 1, 0) * a;
	viewport.pictureRow = Row;
	viewport.pictureCol = Col;
//	viewport.eye       = Vector(0, 0, 2);
//	viewport.upperleft = Vector(-0.8, -0.8, 0.8);
//	viewport.right     = Vector(-0.8, 0.8, 0.8);
//	viewport.bottom    = Vector(0.8, -0.8, 0.8);
//	viewport.pictureRow = Row;
//	viewport.pictureCol = Col;

	scene.light();
	scene.box();
//	scene.crosses();
	scene.tree();

	IplImage* image = cvCreateImage(cvSize(Row, Col), 8, 3);

	int t0 = clock();
	
	for (int i = 0; i < Row; i++) {
		for (int j = 0; j < Col; j++) {
			Color x = viewport.pixel(i, j);
//			if (maxabs(x) > 1) 
//			printf("%lf %lf %lf\n", x.x, x.y, x.z);
			x = clamp(x);
			//printf("%lf %lf %lf\n", x.x, x.y, x.z);

			CV_IMAGE_ELEM(image, char, i, j * 3 + 0) = x.z * 255;
			CV_IMAGE_ELEM(image, char, i, j * 3 + 1) = x.y * 255;
			CV_IMAGE_ELEM(image, char, i, j * 3 + 2) = x.x * 255;
		}
		printf("ROW %d\n", i);
	}

	printf("Time for Rendering = %lfs\n", double(clock() - t0) / CLOCKS_PER_SEC);

	cvSaveImage("scene.png", image);
	cvNamedWindow("scene", CV_WINDOW_AUTOSIZE);
	cvShowImage("scene", image);
	cvWaitKey(0);

	return 0;
}
