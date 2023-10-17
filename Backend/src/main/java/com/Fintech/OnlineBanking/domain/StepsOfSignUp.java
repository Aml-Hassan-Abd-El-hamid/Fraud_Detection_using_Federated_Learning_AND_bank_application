package com.Fintech.OnlineBanking.domain;

import jakarta.persistence.Entity;

import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
@Entity
@Table(name="StepsOfSignUp")
public class StepsOfSignUp {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private Long id;
     private Boolean stepOne=false;
     private Boolean stepTwo=false;
     private Boolean stepThree=false;

     @ManyToOne
 	@JoinColumn(referencedColumnName = "nationalId")
 	
 	private UserInformation userInfo;
     
     
	public UserInformation getUserInfo() {
		return userInfo;
	}
	public void setUserInfo(UserInformation userInfo) {
		this.userInfo = userInfo;
	}
	public Long getId() {
		return id;
	}
	public void setId(Long id) {
		this.id = id;
	}
	public Boolean getStepOne() {
		return stepOne;
	}
	public void setStepOne(Boolean stepOne) {
		this.stepOne = stepOne;
	}
	public Boolean getStepTwo() {
		return stepTwo;
	}
	public void setStepTwo(Boolean stepTwo) {
		this.stepTwo = stepTwo;
	}
	public Boolean getStepThree() {
		return stepThree;
	}
	public void setStepThree(Boolean stepThree) {
		this.stepThree = stepThree;
	}
	
    


}
