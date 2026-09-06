#include <Servo.h>

//  Ultrasonic Sensor
#define TRIG_PIN 8
#define ECHO_PIN 9

//  Servo Motor 
#define SERVO_PIN 10

// L298N Motor Driver 
// Left Motor
#define ENA 5
#define IN1 2
#define IN2 3

// Right Motor
#define ENB 6
#define IN3 4
#define IN4 7

Servo sensorServo;

//Parameters
const int SAFE_DISTANCE = 25;   // Minimum safe distance in cm
const int FORWARD_SPEED = 160;
const int TURN_SPEED = 170;

// Servo positions
const int CENTER = 90;
const int LEFT = 150;
const int RIGHT = 30;



// SETUP

void setup() {

  Serial.begin(9600);

  // Ultrasonic pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // Motor driver pins
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // Servo
  sensorServo.attach(SERVO_PIN);
  sensorServo.write(CENTER);

  stopRobot();

  delay(1000);
}


// MAIN LOOP
void loop() {

  int frontDistance = getDistance();

  Serial.print("Front Distance: ");
  Serial.print(frontDistance);
  Serial.println(" cm");

  // If path is clear, move forward
  if (frontDistance > SAFE_DISTANCE) {

    moveForward(FORWARD_SPEED);
  }

  // If obstacle is detected
  else {

    stopRobot();
    delay(300);

    // Scan left and right
    int leftDistance = scanLeft();
    int rightDistance = scanRight();

    Serial.print("Left Distance: ");
    Serial.print(leftDistance);
    Serial.println(" cm");

    Serial.print("Right Distance: ");
    Serial.print(rightDistance);
    Serial.println(" cm");


    // Return sensor to center
    sensorServo.write(CENTER);
    delay(300);


    // Decide which direction to turn
    if (leftDistance > rightDistance) {

      Serial.println("Turning LEFT");

      turnLeft();
      delay(600);
    }

    else {

      Serial.println("Turning RIGHT");

      turnRight();
      delay(600);
    }

    stopRobot();
    delay(200);
  }
}



// ULTRASONIC DISTANCE MEASUREMENT

int getDistance() {

  long duration;
  int distance;

  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);

  digitalWrite(TRIG_PIN, LOW);

  duration = pulseIn(ECHO_PIN, HIGH, 30000);

  // If no echo is received
  if (duration == 0) {
    return 400;
  }

  distance = duration * 0.034 / 2;

  return distance;
}


// SCAN LEFT

int scanLeft() {

  sensorServo.write(LEFT);
  delay(500);

  int distance = getDistance();

  return distance;
}


// SCAN RIGHT

int scanRight() {

  sensorServo.write(RIGHT);
  delay(500);

  int distance = getDistance();

  return distance;
}

// MOVE FORWARD

void moveForward(int speedValue) {

  // Left motor
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  // Right motor
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);

  analogWrite(ENA, speedValue);
  analogWrite(ENB, speedValue);
}


// MOVE BACKWARD

void moveBackward(int speedValue) {

  // Left motor
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);

  // Right motor
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);

  analogWrite(ENA, speedValue);
  analogWrite(ENB, speedValue);
}


// TURN LEFT

void turnLeft() {

  // Left motor backward
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);

  // Right motor forward
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);

  analogWrite(ENA, TURN_SPEED);
  analogWrite(ENB, TURN_SPEED);
}


// TURN RIGHT

void turnRight() {

  // Left motor forward
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  // Right motor backward
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);

  analogWrite(ENA, TURN_SPEED);
  analogWrite(ENB, TURN_SPEED);
}


// STOP ROBOT

void stopRobot() {

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);

  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
}