package lms;

public class Staff extends User {
    private String position;
    private double salary; 

    public Staff(String position, double salary) { // Modified constructor
        this.position = position;
        this.salary = salary;
    }

    public Staff(String position, double salary, String fullName, int id) { // Modified constructor
        super(fullName, id);
        this.position = position;
        this.salary = salary;
    }

    // Getters and setters for position
    public String getPosition() {
        return position;
    }

    public void setPosition(String position) {
        this.position = position;
    }

    // Getters and setters for salary
    public double getSalary() {
        return salary;
    }

    public void setSalary(double salary) {
        this.salary = salary;
    }

    public String getFullName() {
        return fullName;
    }

    public void setFullName(String fullName) {
        this.fullName = fullName;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }
}
