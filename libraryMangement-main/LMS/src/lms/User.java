
package lms;


public class User {
    protected String fullName;
    protected int id;
    protected String book;

    public User() {
    }

    public User(String fullName) {
        this.fullName = fullName;
    }

    public User(String fullName, int id) {
        this.fullName = fullName;
        this.id = id;
    }

    public User(String fullName, int id,String book) {
        this.fullName = fullName;
        this.id = id;
        this.book = book;
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

    @Override
    public String toString() {
        return "User{" + "fullName=" + fullName + ", id=" + id + ", book=" + book + '}';
    }

}
