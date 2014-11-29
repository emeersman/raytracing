//
//  main.cpp
//  RayTracing
//
//  Created by Emma Meersman on 11/13/14.
//  Copyright (c) 2014 Emma Meersman. All rights reserved.
//

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#include <float.h>
#include "float2.h"
#include "float3.h"
#include "float4.h"
#include "float4x4.h"
#include <vector>
#include <algorithm>
#include "perlin.h"

// For procedural texturing
Perlin perlin;
// for quadrics, so that we do not need a float4.cpp
const float4x4 float4x4::identity(
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1);

// Abstract base class for light sources
class LightSource
{
public:
	virtual float3 getPowerDensityAt(float3 x)=0;
	virtual float3 getLightDirAt(float3 x)=0;
	virtual float  getDistanceFrom(float3 x)=0;

};

class DirectionalLight : public LightSource {
private:

public:
    //DirectionalLight(float3 inputDirection, float3 inputDensity)
    //:direction(inputDirection), density(inputDensity){};
    
    float3 direction;
    float3 density;
    
    // Constant everywhere along the ray
    virtual float3 getPowerDensityAt(float3 x) {
        return density;
    }
    
    // Constant everywhere along the ray
    virtual float3 getLightDirAt (float3 x) {
        return direction.normalize();
    }
    
    virtual float getDistanceFrom (float3 x) {
        return 9001;
    }
};

class PointLight : public LightSource {
public:
    //PointLight(float3 inputPosition, float3 inputPower)
    //:position(inputPosition), power(inputPower){};
    float3 position;
    float3 power;
    
    virtual float3 getPowerDensityAt (float3 x) {
        float surfaceArea = 4*M_PI*pow(getDistanceFrom(x),2.0);
        return float3(power.x/surfaceArea, power.y/surfaceArea, power.z/surfaceArea);
    }
    
    virtual float3 getLightDirAt (float3 x) {
        float3 dirVector = position - x;
        float norm = dirVector.norm();
        return float3(dirVector.x/norm, dirVector.y/norm, dirVector.z/norm);
    }
    
    virtual float getDistanceFrom (float3 x) {
        float3 dirVector = position - x;
        return dirVector.norm();
        //return pow(dirVector.x,2.0) + pow(dirVector.y, 2.0) + pow(dirVector.z, 2.0);
    }
};


// Skeletal Material class. Feel free to add methods e.g. for illumination computation (shading).
class Material
{
public:
	bool reflective;
	bool refractive;
	bool textured;
	float3 minReflectance;		// Fresnel coefficient
	float refractiveIndex;			// index of refraction
	float3 kd;			// diffuse reflection coefficient
	float3 ks;			// specular reflection coefficient
	float shininess;	// specular exponent
	Material()
	{
		reflective = false;
		refractive = false;
		textured = false;
		minReflectance = float3(0.93, 0.85, 0.4);
		refractiveIndex = 1;
		kd = float3(0.5, 0.5, 0.5) + kd * 0.5;
		ks = float3(1, 1, 1);
		shininess = 15;
	}

	float3 reflect(float3 inDir, float3 normal) {
		return inDir - normal * normal.dot(inDir) * 2;
	}
	float3 refract(float3 inDir, float3 normal) {
		float ri = refractiveIndex;
		float cosa = -normal.dot(inDir);
		if(cosa < 0) { cosa = -cosa; normal = -normal; ri = 1 / ri; }
		float disc = 1 - (1 - cosa * cosa) / ri / ri;
		if(disc < 0) return reflect(inDir, normal);
		return inDir * (1.0 / ri) + normal * (cosa / ri - sqrt(disc));
	}
	float3 getReflectance(float3 inDir, float3 normal) {
		float cosa = fabs(normal.dot(inDir));
		return minReflectance + (float3(1, 1, 1) - minReflectance) * pow(1 - cosa, 5);
	}

    virtual float3 shade(float3 normal,
                         float3 viewDir,
                         float3 lightDir,
                         float3 lightPowerDensity)
    {
        float cosTheta = normal.dot(lightDir);
        if(cosTheta < 0) {
            return float3(0,0,0);
        }
        float3 diffuse = lightPowerDensity * kd * cosTheta;

        float3 halfway = (viewDir + lightDir).normalize();
        float cosDelta = normal.dot(halfway);
        if(cosDelta < 0) {
            return diffuse;
        }
        return lightPowerDensity * ks * pow(cosDelta, shininess) + diffuse;
    }
};

// Skeletal Camera class. Feel free to add custom initialization, set aspect ratio to fit viewport dimensions, or animation.
class Camera
{
	float3 eye;

	float3 lookAt;
	float3 right;
	float3 up;

public:
	float3 getEye()
	{
		return eye;
	}
	Camera()
	{
		eye = float3(0, 0, 3);
		lookAt = float3(0, 0, 2);
		right = float3(1, 0, 0);
		up = float3(0, 1, 0);
	}

    void funkyCamera() {
        up = float3(0, 2, -1);
    }
    
	float3 rayDirFromNdc(const float2 ndc) {
		return (lookAt - eye
			+ right * ndc.x
			+ up    * ndc.y
			).normalize();
	}
};

// Ray structure.
class Ray
{
public:
    float3 origin;
    float3 dir;
    Ray(float3 o, float3 d)
    {
        origin = o;
        dir = d;
    }
};

// Hit record structure. Contains all data that describes a ray-object intersection point.
class Hit
{
public:
	Hit()
	{
		t = -1;
	}
	float t;
	float3 position;
	float3 normal;
	Material* material;
};

// Object abstract base class.
class Intersectable
{
protected:
	Material* material;
public:
	Intersectable(Material* material):material(material) {}
    virtual Hit intersect(const Ray& ray)=0;
};

// Object realization.
class Sphere : public Intersectable
{
	float3 center;
	float radius;
public:
    Sphere(const float3& center, float radius, Material* material):
		Intersectable(material),
		center(center),
		radius(radius)
    {}
    
    Hit intersect(const Ray& ray)
    {
        float3 diff = ray.origin - center;
        double a = ray.dir.dot(ray.dir);
        double b = diff.dot(ray.dir) * 2.0;
        double c = diff.dot(diff) - radius * radius;
 
        double discr = b * b - 4.0 * a * c;
        if ( discr < 0 ) 
            return Hit();
        double sqrt_discr = sqrt( discr );
        double t1 = (-b + sqrt_discr)/2.0/a;
        double t2 = (-b - sqrt_discr)/2.0/a;
 
		float t = (t1<t2)?t1:t2;
		if(t < 0)
			t = (t1<t2)?t2:t1;
		if (t < 0)
            return Hit();

		Hit h;
		h.t = t;
		h.material = material;
		h.position = ray.origin + ray.dir * t;
		h.normal = h.position - center;
		h.normal.normalize();

		return h;

    }
}; 

class Plane : public Intersectable
{
    float3 normal;
    float3 x0;
public:
    
    Plane(float3 normal, float3 x0, Material* material):
        Intersectable(material), normal(normal), x0(x0) {}
    
    virtual Hit intersect(const Ray& ray) {
        Hit h;
        h.t = (x0 - ray.origin).dot(normal) / (ray.dir.dot(normal));
        h.material = material;
        h.normal = normal;
        h.position = ray.origin + ray.dir*h.t;
        return h;
    }
};

class Quadric : public Intersectable
{
    float4x4 A;
    char type;
public:
    Quadric(Material* material): Intersectable(material)
    {
        A = float4x4::identity;
    }
    
    void move(float3 diff) {
        float4x4 transform = float4x4::translation(diff).invert();
        
        A = transform * A * transform.transpose();
    }
    
    void scale(float3 diff) {
        float4x4 transform = float4x4::scaling(diff).invert();
        
        A = transform * A * transform.transpose();
    }
    
    void rotate(float3 diff, float angle) {
        float4x4 transform = float4x4::rotation(diff, angle).invert();
        
        A = transform * A * transform.transpose();
    }
    
    void setSphere() {
        A._00 = 1;
        A._11 = 1;
        A._22 = 1;
        A._33 = -1;
    }
    
    void setEllipse() {
        A._00 = 1;
        A._11 = 3;
        A._22 = 4;
        A._33 = -1;
    }
    
    Hit intersect(const Ray& ray) {
        // ray in homo coords
        float4 e = float4(ray.origin.x,
                          ray.origin.y, ray.origin.z, 1);
        float4 d = float4(ray.dir.x,
                          ray.dir.y, ray.dir.z, 0);
        // quadratic coeffs.
        double a = d.dot( A * d );
        double b = e.dot( A * d )
        + d.dot( A * e );
        double c = e.dot( A * e );
        // from here on identical to Sphere
        
        double discr = b * b - 4.0 * a * c;
        if ( discr < 0 )
            return Hit();
        double sqrt_discr = sqrt( discr );
        double t1 = (-b + sqrt_discr)/2.0/a;
        double t2 = (-b - sqrt_discr)/2.0/a;
        
		float t = (t1<t2)?t1:t2;
		if(t < 0)
			t = (t1<t2)?t2:t1;
		if (t < 0)
            return Hit();
        
        Hit h;
        
        h.t = t;
		h.material = material;
		h.position = ray.origin + ray.dir * t;
        
        // homo position
        float4 hPos = float4(h.position.x,
                             h.position.y, h.position.z, 1);
        // homo normal per quadric normal formula
        float4 hNormal = A * hPos +  hPos * A;
        // Cartesian normal
        h.normal = float3(hNormal.x, hNormal.y, hNormal.z).normalize();
        
        h.normal.normalize();
        

        
		return h;
    }

};

class ClippedQuadric : public Intersectable
{
    float4x4 A;
    float4x4 B;
    bool hole;
public:
    ClippedQuadric(Material* material, bool hole)
    :Intersectable(material), hole(hole) {
        // infinite cylinder hardwired
        A = float4x4::identity;
        A._11 = 0;
        A._33 = -1;
        // sphere or radius 2 hardwired
        B = float4x4::identity;
        B._22 = 1;
        B._33 = -2;
    } // add methods to change quadric
    
    void move(float3 diff) {
        float4x4 transform = float4x4::translation(diff).invert();
        
        A = transform * A * transform.transpose();
        B = transform * B * transform.transpose();
    }
    
    void scale(float3 diff) {
        float4x4 transform = float4x4::scaling(diff).invert();
        
        A = transform * A * transform.transpose();
        B = transform * B * transform.transpose();
    }
    
    void rotate(float3 diff, float angle) {
        float4x4 transform = float4x4::rotation(diff, angle);
        
        A = transform * A * transform.transpose();
        B = transform * B * transform.transpose();
    }

    
    void setSphere() {
        A._00 = 1;
        A._11 = 1;
        A._22 = 1;
        A._33 = -1;
    }
    
    void setCone() {
        A._00 = 0.1;
        A._11 = 8;
        A._22 = 8;
        A._33 = -1;
        
        B._00 = 2;
        B._11 = 4;
        B._22 = 4;
        B._33 = -1;
        
        float4x4 transform = float4x4::translation(float3(3,0,0)).invert();
        B = transform * B * transform.transpose();
    }
    
    void setEllipse() {
        A._00 = 1;
        A._11 = 3;
        A._22 = 4;
        A._33 = -1;
    }
    
    void setHole(float3 scale, float3 translation, float3 rotation, float rotDegree) {
        B._00 = 1;
        B._11 = 0;
        B._22 = 1;
        B._33 = -1;
        
        float4x4 transform = float4x4::scaling(scale).invert();

        B = transform * B * transform.transpose();
        
        float4x4 translate = float4x4::translation(translation).invert();
        
        B = translate * B * translate.transpose();
        
        float4x4 rotate = float4x4::rotation(rotation, rotDegree);
        
        B = rotate * B * rotate.transpose();
        
    }
    
    Hit intersect(const Ray& ray) {
        // ray in homo coords
        float4 e = float4(ray.origin.x,
                          ray.origin.y, ray.origin.z, 1);
        float4 d = float4(ray.dir.x,
                          ray.dir.y, ray.dir.z, 0);
        // quadratic coeffs.
        double a = d.dot( A * d );
        double b = e.dot( A * d )
        + d.dot( A * e );
        double c = e.dot( A * e );
        // from here on identical to Sphere
        
        double discr = b * b - 4.0 * a * c;
        if ( discr < 0 )
            return Hit();
        double sqrt_discr = sqrt( discr );
        double t1 = (-b + sqrt_discr)/2.0/a;
        double t2 = (-b - sqrt_discr)/2.0/a;
        
        if(!hole){
        float4 hit1 = e + d * t1;
        if(hit1.dot(B * hit1) > 0) // if not in B
            t1 = -1;				 // invalidate
        float4 hit2 = e + d * t2;
        if(hit2.dot(B * hit2) > 0) // if not in B
            t2 = -1; 				 // invalidate
        } else {
            float4 hit1 = e + d * t1;
            if(hit1.dot(B * hit1) <= 0) // if not in B
                t1 = -1;				 // invalidate
            float4 hit2 = e + d * t2;
            if(hit2.dot(B * hit2) <= 0) // if not in B
                t2 = -1; 				 // invalidate
            
            float t = (t1<t2)?t1:t2;
            if(t < 0)
                t = (t1<t2)?t2:t1;
            if (t < 0)
                return Hit();
        }
        
        float t = (t1<t2)?t1:t2;
		if(t < 0)
			t = (t1<t2)?t2:t1;
		if (t < 0)
            return Hit();
    
        Hit h;
        
        h.t = t;
		h.material = material;
		h.position = ray.origin + ray.dir * t;
        
        // homo position
        float4 hPos = float4(h.position.x,
                             h.position.y, h.position.z, 1);
        // homo normal per quadric normal formula
        float4 hNormal = A * hPos +  hPos * A;
        // Cartesian normal
        h.normal = float3(hNormal.x, hNormal.y, hNormal.z).normalize();
        
        h.normal.normalize();
        
		return h;
    }

};

class Scene
{
	Camera camera;
	std::vector<LightSource*> lightSources;
	std::vector<Intersectable*> objects;
	std::vector<Material*> materials;
    
public:
	Scene()
	{
        /* Light Sources */
        DirectionalLight *dir = new DirectionalLight();
        dir->direction = float3(1,1,0.9);
        dir->density = float3(0.95, 0.95, 0.95);
        
        PointLight *point = new PointLight();
        point->position = float3(1,3,3);
        point->power = float3(250,250,250);
        
        PointLight *point2 = new PointLight();
        point2->position = float3(-2,4,2);
        point2->power = float3(0,0,255);
        
		lightSources.push_back(dir);
        //lightSources.push_back(point);
        lightSources.push_back(point2);
        
        
        /* Materials in the Scene */
        
        Material *shiny = new Material();
        shiny->shininess = 30;
        shiny->ks = float3(0.2,0.2,0.3);
        shiny->kd = float3(0.95,0.95,1.1);
        materials.push_back(shiny);
        
        Material *snow = new Material();
        snow->ks = float3(1, 1, 1);
        snow->kd = float3(0.32, 0.32, 0.32);
        snow->shininess = 1;
        materials.push_back(snow);
        
        Material *copper = new Material();
        copper->ks = float3(1,1,1);
        copper->kd = float3(1, 0.3, 0);
        copper->shininess = 30;
        copper->reflective = true;
        materials.push_back(copper);
        
        Material *carrot = new Material();
        carrot->ks = float3(0.2,0.2,0);
        carrot->kd = float3(1, 0.4, 0);
        carrot->shininess = 0;
        materials.push_back(carrot);
        
        Material *ice = new Material();
        ice->ks = float3(1,1,1);
        ice->kd = float3(1, 1, 1);
        ice->shininess = 10;
        ice->refractive = true;
        ice->reflective = true;
        materials.push_back(ice);
        
        Material *glass = new Material();
        glass->ks = float3(1,1,1);
        glass->kd = float3(1, 1, 1);
        glass->shininess = 50;
        glass->reflective = true;
        materials.push_back(glass);
        
        /* Background */
        
        Plane *p = new Plane(float3(0,1,0), float3(-1,-2,-1), snow);
        objects.push_back(p);
        
        ClippedQuadric *icicle = new ClippedQuadric(ice, false);
        icicle->setCone();
        icicle->scale(float3(1.8,0.5,0.5));
        icicle->rotate(float3(0,0,1),-1.2);
        icicle->move(float3(-0.8,-4,0.3));
        objects.push_back(icicle);
        
        ClippedQuadric *icicle2 = new ClippedQuadric(ice, false);
        icicle2->setCone();
        icicle2->scale(float3(1.8,0.5,0.5));
        icicle2->rotate(float3(0,0,1),-1.2);
        icicle2->rotate(float3(0,1,0),-1);
        icicle2->move(float3(-1.6,-4,1.26));
        objects.push_back(icicle2);
        
        /* Snowman and associated objects */
        
        Sphere *eye1 = new Sphere(float3(0.15,0.88,0.5),0.05,glass);
        objects.push_back(eye1);
        
        Sphere *eye2 = new Sphere(float3(-0.2,0.89,0.48),0.03,glass);
        objects.push_back(eye2);
        
        ClippedQuadric *pot = new ClippedQuadric(copper, false);
        pot->scale(float3(0.4,0.25,0.25));
        pot->move(float3(0.05,1.22,0.25));
        pot->rotate(float3(0.3,0,0.7), 0.1);
        objects.push_back(pot);
        
        ClippedQuadric *carrotNose = new ClippedQuadric(carrot, false);
        carrotNose->setCone();
        carrotNose->scale(float3(0.3,0.3,0.3));
        carrotNose->move(float3(-0.2,0.75,-0.15));
        carrotNose->rotate(float3(0,1,0),2);
        objects.push_back(carrotNose);
        
        ClippedQuadric *top = new ClippedQuadric(shiny, false);
        top->setEllipse();
        top->scale(float3(0.6,0.6,0.6));
        top->move(float3(0.1,0.75,0.2));
        objects.push_back(top);
        
        ClippedQuadric *middle = new ClippedQuadric(shiny, true);
        middle->setEllipse();
        middle->setHole(float3(0.3,0.3,0.3), float3(0,0,0), float3(1,0,0), 1.7);
        middle->rotate(float3(0,0.2,0),0.2);
        objects.push_back(middle);
        
        Quadric *bottom = new Quadric(shiny);
        bottom->setEllipse();
        bottom->scale(float3(1.5,1.5,1.5));
        bottom->move(float3(0,-1.2,0));
        bottom->rotate(float3(0,0.2,0),-0.2);
        objects.push_back(bottom);
        
        /* UFO reflected in snowman's hat */

        Quadric *ufo = new Quadric(shiny);
        ufo->setEllipse();
        ufo->scale(float3(2,0.1,0.1));
        ufo->rotate(float3(0,0,1), -0.2);
        ufo->move(float3(0.5,3.7,5));
        objects.push_back(ufo);
        
        Sphere *ufoBody = new Sphere(float3(0.5,3.7,5), 0.4, shiny);
        objects.push_back(ufoBody);
        

	}
	~Scene()
	{
		for (std::vector<LightSource*>::iterator iLightSource = lightSources.begin(); iLightSource != lightSources.end(); ++iLightSource)
			delete *iLightSource;
		for (std::vector<Material*>::iterator iMaterial = materials.begin(); iMaterial != materials.end(); ++iMaterial)
			delete *iMaterial;
		for (std::vector<Intersectable*>::iterator iObject = objects.begin(); iObject != objects.end(); ++iObject)
			delete *iObject;		
	}

public:
	Camera& getCamera()
	{
		return camera;
	}

	Hit firstIntersect(const Ray& ray)
	{
		Hit bestHit;
		bestHit.t = FLT_MAX;
		for(int oi=0; oi < objects.size(); oi++)
		{
			Hit hit = objects[oi]->intersect(ray);
			if(hit.t > 0 && hit.t < bestHit.t)
				bestHit = hit;
		}
		if(bestHit.t == FLT_MAX)
			return Hit();
		return bestHit;
	}

	float3 trace(const Ray& ray, int depth)
	{
        Hit hit = firstIntersect(ray);

        // If ray hits nothing, we return the aurora borealis
		if(hit.t < 0) {
            return ray.dir * ray.dir * float3(0,1,0.6);
        }
        
        float3 lightSum = {0,0,0};

        for(LightSource *l : lightSources) {
            float3 li = l->getLightDirAt(ray.origin);
            float dist = l->getDistanceFrom(hit.position);
            int maxDepth = 5;

            // Deals with shadows
            Ray shadowRay(hit.position + (hit.normal*0.1), li);
            Hit shadowHit = firstIntersect(shadowRay);
            if(shadowHit.t > 0 && shadowHit.t < dist)
                continue;
            
            // Handles types of materials differently
            if(depth > maxDepth) return lightSum;
            if(hit.material->reflective){   // for smooth surface
                float3 reflectionDir = hit.material->reflect(ray.dir, hit.normal);
                Ray reflectedRay(hit.position + hit.normal*0.1, reflectionDir );
                lightSum += trace(reflectedRay, depth+1)
                        * hit.material->getReflectance(ray.dir, hit.normal);
            }
            if(hit.material->refractive) {  // for smooth surface
                float3 refractionDir = hit.material->refract(ray.dir, hit.normal);
                Ray refractedRay(hit.position - hit.normal*0.1, refractionDir );
                lightSum += trace(refractedRay, depth+1) * (float3(1,1,1) - hit.material->getReflectance(ray.dir, hit.normal));
            } else {                        // for rough surface
                lightSum += hit.material->shade(hit.normal, -ray.dir, l->getLightDirAt(hit.position),
                                           l->getPowerDensityAt(hit.position));
            }
        }
        
		return lightSum;
	}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////
// global application data

// screen resolution
const int screenWidth = 600;
const int screenHeight = 600;
// image to be computed by ray tracing
float3 image[screenWidth*screenHeight];

Scene scene;

bool computeImage()
{
	static unsigned int iPart = 0;

	if(iPart >= 64)
		return false;
    for(int j = iPart; j < screenHeight; j+=64)
	{
        for(int i = 0; i < screenWidth; i++)
		{
			float3 pixelColor = float3(0, 0, 0);
			float2 ndcPixelCentre( (2.0 * i - screenWidth) / screenWidth, (2.0 * j - screenHeight) / screenHeight );

			Camera& camera = scene.getCamera();
			Ray ray = Ray(camera.getEye(), camera.rayDirFromNdc(ndcPixelCentre));
			
			image[j*screenWidth + i] = scene.trace(ray, 0);
		}
	}
	iPart++;
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL starts here. In the ray tracing example, OpenGL just outputs the image computed to the array.

// display callback invoked when window needs to be redrawn
void onDisplay( ) {
    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear screen

	if(computeImage())
		glutPostRedisplay();
    glDrawPixels(screenWidth, screenHeight, GL_RGB, GL_FLOAT, image);
 
    glutSwapBuffers(); // drawing finished
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);						// initialize GLUT
    glutInitWindowSize(600, 600);				// startup window size 
    glutInitWindowPosition(100, 100);           // where to put window on screen
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);    // 8 bit R,G,B,A + double buffer + depth buffer
 
    glutCreateWindow("Ray caster");				// application window is created and displayed
 
    glViewport(0, 0, screenWidth, screenHeight);

    glutDisplayFunc(onDisplay);					// register callback
 
    glutMainLoop();								// launch event handling loop
    
    return 0;
}

