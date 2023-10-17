package com.Fintech.OnlineBanking.repository;

import java.util.Collection;


import org.springframework.data.jpa.repository.JpaRepository;

import com.Fintech.OnlineBanking.domain.StepsOfSignUp;
import com.Fintech.OnlineBanking.domain.UserAccounts;

public interface StepsOfSignUpRepository extends JpaRepository <StepsOfSignUp, Long>{
	StepsOfSignUp findByUserInfoNationalId(Long nationalId);
}
