package lms;
public class Student extends User {
    private int year;
    private String department ;
    
    
    public Student() {
    }
    
    public Student(int year, String department, String fullname) {
        super(fullname);
        this.year = year;
        this.department = department;
    }

    public Student(int year, int id,String department, String fullName) {
        super(fullName, id);
        this.year = year;
        this.department = department;
    }
    public Student(int year, int id,String department, String fullName,String book) {
        super(fullName, id,book);
        this.year = year;
        this.department = department;
    }
    
    public int getYear() {
        return year;
    }

    public void setYear(int year) {
        this.year = year;
    }

    public String getDepartment() {
        return department;
    }

    public void setDepartment(String department) {
        this.department = department;
    }

    public String getFullName() {
        return fullName;
    }

    public void setFullName(String fullName) {
        this.fullName = fullName;
    }

    @Override
    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getBook() {
        return book;
    }

    public void setBook(String book) {
        this.book = book;
    }

    @Override
    public String toString() {
        return super.toString()+"Student{" + "year=" + year + ", department=" + department + '}';
    }

    
    
}
