package com.Fintech.OnlineBanking.dto;

public class passwordRequest {
   /* @Pattern(regexp = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[@$!%*?&])[A-Za-z\\d@$!%*?&]{8,}$"
    		,message="Minimum eight characters, at least one uppercase letter, one lowercase letter, one number and one special character")
*/
	private String password;
    /*@Transient
    @Pattern(regexp = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[@$!%*?&])[A-Za-z\\d@$!%*?&]{8,}$"
	,message="confirm password Minimum eight characters, at least one uppercase letter, one lowercase letter, one number and one special character")

	private String confirmPassword;
    */
    /*public String c() {
    if(this.Confirmpassword!=this.password) {
    	return"notmatch";
    }
	return Confirmpassword;}*/
  
	private Long nationalId ;




	
	/*public String getConfirmPassword() {
		return confirmPassword;
	}

	public void setConfirmPassword(String confirmPassword) {
		this.confirmPassword = confirmPassword;
	}*/

	public Long getNationalId() {
		return nationalId;
	}

	public void setNationalId(Long nationalId) {
		this.nationalId = nationalId;
	}

	public String getPassword() {
		return password;
	}

	public void setPassword(String password) {
		this.password = password;
	}

}
